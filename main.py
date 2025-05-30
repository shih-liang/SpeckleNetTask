import os
from pathlib import Path
from typing import Dict, Any
import torch
import torch.optim as optim
import torch.utils.data as data
import yaml
from tqdm import tqdm
import wandb

from utils.summary import print_model_summary
from datasets.paired import create_train_dataset, create_val_dataset
from losses import TVLoss, L1Loss
from models import DenoisingNet
from metrics import calculate_batch_psnr

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters"""
    required_sections = ['data', 'model', 'training', 'loss']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config")
    
    # Validate data section
    data_config = config['data']
    required_data = ['train_dir', 'val_dir', 'intervals']
    for param in required_data:
        if param not in data_config:
            raise ValueError(f"Missing required parameter '{param}' in data section")
    
    # Validate val_dir structure
    val_dir = data_config['val_dir']
    if not isinstance(val_dir, dict) or 'noise' not in val_dir or 'refer' not in val_dir:
        raise ValueError("val_dir must be a dictionary with 'noise' and 'refer' keys")
    
    # Validate training section
    training_config = config['training']
    required_training = ['device', 'batch_size', 'num_workers', 'learning_rate', 
                        'epochs', 'save_interval', 'checkpoint_dir']
    for param in required_training:
        if param not in training_config:
            raise ValueError(f"Missing required parameter '{param}' in training section")
    
    # Validate model section
    model_config = config['model']
    required_model = ['in_channels', 'out_channels']
    for param in required_model:
        if param not in model_config:
            raise ValueError(f"Missing required parameter '{param}' in model section")
    
    # Validate loss section
    loss_config = config['loss']
    required_loss = ['tv_weight']
    for param in required_loss:
        if param not in loss_config:
            raise ValueError(f"Missing required parameter '{param}' in loss section")
    
    # Validate wandb section if present
    if 'wandb' in config:
        wandb_config = config['wandb']
        required_wandb = ['project', 'log_interval']
        for param in required_wandb:
            if param not in wandb_config:
                raise ValueError(f"Missing required parameter '{param}' in wandb section")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file with environment variable support"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables
    def replace_env_vars(value: Any) -> Any:
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var)
        return value
    
    def process_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: process_value(v) for k, v in d.items()}
    
    def process_value(v: Any) -> Any:
        if isinstance(v, dict):
            return process_dict(v)
        elif isinstance(v, list):
            return [process_value(item) for item in v]
        else:
            return replace_env_vars(v)
    
    config = process_dict(config)
    validate_config(config)
    return config

def setup_directories(config):
    """Create necessary directories"""
    # Create checkpoint directory
    Path(config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
def create_dataloaders(config):
    """Create and return dataloaders"""
    # Create training dataset
    train_dataset = create_train_dataset(
        config['data']['train_dir'],
        config['data']['intervals']
    )
    
    # Create validation dataset
    val_dataset = create_val_dataset(
        config['data']['val_dir']['noise'],
        config['data']['val_dir']['refer']
    )
    
    # Create dataloaders with memory optimizations
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=1,  # Number of batches loaded in advance by each worker
        drop_last=True  # Drop incomplete batches to avoid memory issues
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
        drop_last=True
    )
    
    return train_loader, val_loader

def create_model(config):
    """Create and return model"""
    device = torch.device(config['training']['device'])
    model = DenoisingNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        hidden_channels=16,
        scales=[3,5,7,13,17]
    )
    model = model.to(device)  # Move model to the specified device
    if config['training']['parallel_train']:
        model = torch.nn.DataParallel(model)
    print_model_summary(model, device, (config['model']['in_channels'], 512, 512))
    return model

def create_optimizer(model, config):
    """Create and return optimizer"""
    return optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

def create_loss_functions(config):
    """Create and return loss functions"""
    device = torch.device(config['training']['device'])
    return {
        'l1': L1Loss(reduction='mean').to(device),
        'tv': TVLoss(reduction='mean').to(device)
    }

def init_wandb(config: Dict[str, Any]) -> bool:
    """Initialize Weights & Biases"""
    try:
        wandb.init(
            project=config['wandb']['project'],
            config=config,
            name=config['wandb'].get('name', None),
            tags=config['wandb'].get('tags', []),
            notes=config['wandb'].get('notes', '')
        )
        return True
    except ImportError:
        print("Warning: wandb not installed. Running without experiment tracking.")
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
    return False

def evaluate_model(model, val_loader, loss_functions, device, config):
    """Evaluate model performance by computing various metrics"""
    model.eval()
    total_loss = 0
    total_l1_loss = 0
    total_tv_loss = 0
    total_psnr = 0
    total_psnr_std = 0
    
    with torch.no_grad():
        for lr_data, hr_data in val_loader:
            # Move data to device
            lr_data = lr_data.to(device, non_blocking=True)
            hr_data = hr_data.to(device, non_blocking=True)
            
            # Forward pass
            output = model(lr_data)
            
            # Calculate losses in log space
            hr_log = torch.log1p(hr_data)
            l1_loss = loss_functions['l1'](output, hr_log)
            tv_loss = loss_functions['tv'](output)
            loss = l1_loss + config['loss']['tv_weight'] * tv_loss
            
            # Calculate PSNR in original space
            exp_output = torch.expm1(output)
            batch_psnr_mean, batch_psnr_std = calculate_batch_psnr(exp_output, hr_data)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_l1_loss += l1_loss.item()
            total_tv_loss += tv_loss.item()
            total_psnr += batch_psnr_mean
            total_psnr_std += batch_psnr_std
    
    # Calculate average metrics
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_l1_loss = total_l1_loss / num_batches
    avg_tv_loss = total_tv_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_psnr_std = total_psnr_std / num_batches
    
    return {
        'val_loss': avg_loss,
        'val_l1_loss': avg_l1_loss,
        'val_tv_loss': avg_tv_loss,
        'val_psnr': avg_psnr,
        'val_psnr_std': avg_psnr_std
    }

def train_epoch(model, train_loader, optimizer, loss_functions, device, epoch, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_l1_loss = 0
    total_tv_loss = 0
    
    # Enable memory efficient attention if available
    if hasattr(torch, 'set_grad_enabled'):
        torch.set_grad_enabled(True)
    
    with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
        for batch_idx, (lr_data, hr_data) in enumerate(pbar):
            # Clear GPU cache periodically
            if batch_idx % 1 == 0:
                torch.cuda.empty_cache()
            
            # Move data to device
            lr_data = lr_data.to(device, non_blocking=True)
            hr_data = hr_data.to(device, non_blocking=True)
            
            # Debug input data
            if torch.isnan(lr_data).any() or torch.isnan(hr_data).any():
                continue
            
            # Forward pass
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            output = model(lr_data)
            
            # Calculate losses in log space
            hr_log = torch.log1p(hr_data)
            
            # Debug log transform
            if torch.isnan(hr_log).any():
                continue
            
            l1_loss = loss_functions['l1'](output, hr_log)
            tv_loss = loss_functions['tv'](output)
            
            # Debug individual losses
            if torch.isnan(l1_loss).any():
                continue
                
            if torch.isnan(tv_loss).any():
                continue
            
            loss = l1_loss + config['loss']['tv_weight'] * tv_loss
            
            # Debug final loss
            if torch.isnan(loss).any():
                continue
            
            # Backward pass
            loss.backward()

            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            total_l1_loss += l1_loss.item()
            total_tv_loss += tv_loss.item()
            
            current_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': current_loss})
            
            # Log batch metrics if wandb is available
            if 'wandb' in config and batch_idx % config['wandb']['log_interval'] == 0:
                try:
                    wandb.log({
                        'batch/loss': loss.item(),
                        'batch/l1_loss': l1_loss.item(),
                        'batch/tv_loss': tv_loss.item(),
                        'batch/learning_rate': optimizer.param_groups[0]['lr']
                    })
                except Exception as e:
                    print(f'Wandb logging failed: {e}')
                    pass
            
            # Clear some memory after logging
            del output, l1_loss, tv_loss, loss
            torch.cuda.empty_cache()
    
    # Calculate average losses
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_l1_loss = total_l1_loss / num_batches
    avg_tv_loss = total_tv_loss / num_batches
    
    return {
        'train_loss': avg_loss,
        'train_l1_loss': avg_l1_loss,
        'train_tv_loss': avg_tv_loss
    }

def save_checkpoint(model, optimizer, epoch, loss, config):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    checkpoint_path = Path(config['training']['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Initialize wandb (optional)
    wandb_initialized = init_wandb(config)
    
    # Setup
    setup_directories(config)
    train_loader, val_loader = create_dataloaders(config)
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    loss_functions = create_loss_functions(config)
    if wandb_initialized:
        wandb.watch(model, log="all")
    
    # Training loop
    best_val_loss = float('inf')
    best_val_psnr = 0.0  # Track best PSNR
    for epoch in range(config['training']['epochs']):
        # Train epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_functions,
            config['training']['device'], epoch, config
        )
        
        # Evaluate model
        val_metrics = evaluate_model(
            model, val_loader, loss_functions,
            config['training']['device'], config
        )
        
        # Log metrics if wandb is available
        if wandb_initialized:
            log = {
                    'epoch': epoch,
                    **train_metrics,
                    **val_metrics,
                    'epoch/learning_rate': optimizer.param_groups[0]['lr']
                }
            try:
                wandb.log(log)
            except Exception as e:
                print(f'Wandb logging {log} failed: {e}')
                pass
        
        # Save checkpoint if validation metrics improved
        if val_metrics['val_loss'] < best_val_loss or val_metrics['val_psnr'] > best_val_psnr:
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
            if val_metrics['val_psnr'] > best_val_psnr:
                best_val_psnr = val_metrics['val_psnr']
            
            save_checkpoint(model, optimizer, epoch, val_metrics['val_loss'], config)
            if wandb_initialized:
                try:
                    wandb.save(str(Path(config['training']['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pt'))
                except Exception as e:
                    print('Wandb save failed: ', e)
                    pass
        
        # Print epoch results
        print(f'Epoch {epoch + 1}/{config["training"]["epochs"]}')
        print(f'Train Loss: {train_metrics["train_loss"]:.4f}, Val Loss: {val_metrics["val_loss"]:.4f}')
        print(f'Val PSNR: {val_metrics["val_psnr"]:.2f} dB')
    
    # Close wandb if it was initialized
    if wandb_initialized:
        try:
            wandb.finish()
        except Exception as e:
            print('Wandb finish failed: ', e)
            pass

if __name__ == '__main__':
    main() 
