"""Distributed Data Parallel (DDP) utilities for training.

This module provides utilities for setting up and managing distributed training
using PyTorch's DistributedDataParallel.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

def setup_distributed(rank: int, world_size: int) -> None:
    """Initialize distributed training environment.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup_distributed() -> None:
    """Cleanup distributed training environment."""
    dist.destroy_process_group()

def create_distributed_samplers(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    rank: int,
    world_size: int
) -> Tuple[DistributedSampler, DistributedSampler]:
    """Create distributed samplers for training and validation datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        rank: Process rank
        world_size: Total number of processes
        
    Returns:
        Tuple of (train_sampler, val_sampler)
    """
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank
    )
    return train_sampler, val_sampler

def wrap_model_ddp(
    model: torch.nn.Module,
    rank: int
) -> DDP:
    """Wrap model with DistributedDataParallel.
    
    Args:
        model: PyTorch model to wrap
        rank: Process rank
        
    Returns:
        DDP-wrapped model
    """
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    return DDP(model, device_ids=[rank])

def save_checkpoint_ddp(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    rank: int
) -> Optional[Path]:
    """Save model checkpoint in distributed setting.
    
    Args:
        model: DDP-wrapped model
        optimizer: Model optimizer
        epoch: Current epoch
        loss: Current loss value
        config: Training configuration
        rank: Process rank
        
    Returns:
        Path to saved checkpoint if rank is 0, None otherwise
    """
    if rank == 0:  # Only save on main process
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),  # Save DDP model state
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        
        checkpoint_path = Path(config['training']['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    return None

def setup_ddp_training(
    train_fn,
    config: Dict[str, Any]
) -> Tuple[int, mp.Process]:
    """Setup distributed training environment.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (world_size, training_process)
    """
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available")
    
    # Start distributed training
    process = mp.spawn(
        train_fn,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )
    
    return world_size, process

def is_main_process(rank: int) -> bool:
    """Check if current process is the main process.
    
    Args:
        rank: Process rank
        
    Returns:
        True if process is main process (rank 0), False otherwise
    """
    return rank == 0

def setup_ddp(config: Dict[str, Any]) -> Tuple[int, mp.Process]:
    """Alias for setup_ddp_training for backward compatibility.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (world_size, training_process)
    """
    return setup_ddp_training(config) 