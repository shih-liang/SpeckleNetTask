"""Main training script for DWConv model.

This script handles the training process using distributed data parallelism.
"""

import os
import torch
import torch.multiprocessing as mp
from typing import Dict, Any
import argparse
from tqdm import tqdm
from metrics.psnr import calculate_batch_psnr
import json
import matplotlib.pyplot as plt
import numpy as np

from utils.config import Config
from utils.setup import TrainingSetup, ValidationSetup


def _validate(model, dataloader, criterion, device, rank: int, im_output=False):
    try:
        # Run testing
        model.eval()
        total_loss = 0
        total_psnr = 0
        total_psnr_std = 0
        num_batches = 0
        
        # Save all outputs and metrics
        if im_output:
            all_outputs = []
            all_targets = []
        all_metrics = []
        
        with torch.no_grad():
            with tqdm(dataloader, desc='Validation',
                     disable=rank != 0) as pbar:
                for batch_idx, (noise, target) in enumerate(pbar):
                    noise, target = noise.to(device), target.to(device)
                    
                    # Forward pass
                    output = model(noise)
                    target_log = torch.log1p(target)
                    loss = criterion(output, target_log)
                    
                    # Calculate PSNR
                    # Convert from log space to original space for PSNR calculation
                    output_exp = torch.expm1(output)
                    batch_psnr_mean, batch_psnr_std = calculate_batch_psnr(output_exp, target)
                    
                    # Update metrics
                    total_loss += loss.item()
                    total_psnr += batch_psnr_mean
                    total_psnr_std += batch_psnr_std
                    num_batches += 1
                    
                    # Store outputs and targets
                    if im_output:
                        all_outputs.append(output_exp.cpu())
                        all_targets.append(target.cpu())

                    all_metrics.append({
                        'batch_idx': batch_idx,
                        'loss': loss.item(),
                        'psnr': batch_psnr_mean,
                        'psnr_std': batch_psnr_std
                    })
                    
                    if rank == 0:
                        pbar.set_postfix({
                            'loss': loss.item(),
                            'psnr': batch_psnr_mean
                        })
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
        avg_psnr_std = total_psnr_std / num_batches

        metrics = {
            'average_metrics': {
                'loss': avg_loss,
                'psnr': avg_psnr,
                'psnr_std': avg_psnr_std
            },
            'per_batch_metrics': all_metrics
        }
        
        if im_output:
            return metrics, all_outputs, all_targets
        else:
            return metrics
    
    except Exception as e:
        print(f"Error in validation process {rank}: {str(e)}")
        raise e


def train(rank: int, world_size: int, config: Dict[str, Any]) -> None:
    """Training process for a single GPU"""
    setup = None
    try:
        # Initialize training setup
        setup = TrainingSetup(config, rank, world_size)
    
        # Training loop
        for epoch in range(config['training']['epochs']):
            if world_size > 1:
                setup.train_loader.sampler.set_epoch(epoch)
    
            # Training phase
            setup.model.train()
            total_loss = 0
            
            with tqdm(setup.train_loader, desc=f'Epoch {epoch + 1}/{config["training"]["epochs"]} [Train]',
                     disable=rank != 0) as pbar:
                for batch_idx, (noise, target) in enumerate(pbar):
                    noise, target = noise.to(setup.device), target.to(setup.device)
    
                    # Forward pass
                    setup.optimizer.zero_grad()
                    output = setup.model(noise)
                    loss = setup.criterion(output, target)
    
                    # Backward pass
                    loss.backward()
                    setup.optimizer.step()
    
                    # Update metrics
                    total_loss += loss.item()
                    if rank == 0:
                        pbar.set_postfix({'loss': loss.item()})
            
                    if batch_idx % config['training']['val_interval'] == 0:
                        # Validation phase
                        metrics = _validate(setup.model, setup.val_loader, setup.criterion, setup.device, rank, im_output=False)
                
                        # Log metrics
                        if rank == 0:
                            if setup.wandb is not None:
                                setup.wandb.log({
                                    'epoch': epoch,
                                    'train_loss': total_loss / len(setup.train_loader),
                                    'val_loss': metrics['average_metrics']['loss']
                                })
                
            # Save checkpoint
            if rank == 0 and (epoch + 1) % config['training']['save_interval'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': setup.model.module.state_dict() if isinstance(setup.model, torch.nn.parallel.DistributedDataParallel) else setup.model.state_dict(),
                    'optimizer_state_dict': setup.optimizer.state_dict(),
                    'config': config
                }
                
                checkpoint_path = os.path.join(
                    config['training']['checkpoint_dir'],
                    f'checkpoint_epoch_{epoch + 1}.pt'
                )
                
                torch.save(checkpoint, checkpoint_path)
                
                print(f'Epoch {epoch + 1}/{config["training"]["epochs"]}')
                print(f'Train Loss: {total_loss / len(setup.train_loader):.4f}, Val Loss: {metrics["average_metrics"]["loss"]:.4f}')
        
    except Exception as e:
        print(f"Error in training process {rank}: {str(e)}")
        raise e
    finally:
        # Cleanup
        if setup is not None:
            setup.cleanup()

def test(rank: int, world_size: int, config: Dict[str, Any], checkpoint_path: str) -> None:
    """Test process for a single GPU"""
    setup = None
    try:
        # Initialize validation setup
        setup = ValidationSetup(config, rank, world_size)
        
        # Load checkpoint
        if rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
        setup.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create output directories if they don't exist
        output_dir = os.path.join(config['training']['checkpoint_dir'], 'test_outputs')
        images_dir = os.path.join(output_dir, 'images')
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
        
        metrics, all_outputs, all_targets = _validate(setup.model, setup.val_loader, setup.criterion, setup.device, rank, im_output=True)
        
        # Save results on rank 0
        if rank == 0:
            # Save metrics to JSON
            metrics_path = os.path.join(output_dir, 'test_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Save model outputs and targets
            for batch_idx, (output_imgs, target_imgs) in enumerate(zip(all_outputs, all_targets)):
                batch_size = output_imgs.size(0)
                for img_idx in range(batch_size):
                    # Get the output and target images
                    output_img = output_imgs[img_idx].cpu()
                    target_img = target_imgs[img_idx].cpu()
                            
                    # Create a figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    
                    # Plot output image
                    ax1.imshow(output_img.permute(1, 2, 0).numpy())
                    ax1.set_title('Model Output')
                    ax1.axis('off')
                    
                    # Plot target image
                    ax2.imshow(target_img.permute(1, 2, 0).numpy())
                    ax2.set_title('Target')
                    ax2.axis('off')
                    
                    # Save the figure
                    img_path = os.path.join(images_dir, f'batch_{batch_idx}_img_{img_idx}.png')
                    plt.savefig(img_path, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
            
            print("\nTest Results:")
            print(f"Validation Loss: {metrics['average_metrics']['loss']:.4f}")
            print(f"PSNR: {metrics['average_metrics']['psnr']:.2f} ± {metrics['average_metrics']['psnr_std']:.2f} dB")
            print(f"\nResults saved to {output_dir}")
            
    except Exception as e:
        print(f"Error in test process {rank}: {str(e)}")
        raise e
    finally:
        # Cleanup
        if setup is not None:
            setup.cleanup()

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or test the DWConv model')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                      help='Mode to run the script in (train or test)')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint file for testing mode')
    args = parser.parse_args()
    
    # Load configuration
    config_manager = Config('config.yaml')
    config = config_manager.config
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    
    if args.mode == 'train':
        if world_size > 1:
            # Use distributed training
            mp.spawn(
                train,
                args=(world_size, config),
                nprocs=world_size,
                join=True
            )
        else:
            # Single GPU training
            train(0, 1, config)
    else:  # test mode
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for test mode")
            
        if world_size > 1:
            # Use distributed testing
            mp.spawn(
                test,
                args=(world_size, config, args.checkpoint),
                nprocs=world_size,
                join=True
            )
        else:
            # Single GPU testing
            test(0, 1, config, args.checkpoint)

if __name__ == '__main__':
    main() 
