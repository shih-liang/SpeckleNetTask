"""Training setup manager.

This module handles the setup of training components including data loaders,
model, optimizer, and loss functions.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Tuple, Dict, Any, Optional

from utils.ddp_utils import setup_ddp
from utils.wandb_wrapper import WandbWrapper
from models.denoising import DenoisingNet
from datasets.paired import create_train_dataset, create_val_dataset
from losses import TVLoss, L1Loss

class BaseSetup:
    """Base class for training and validation setup"""
    def __init__(self, config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(config['training']['device'])
        
        # Setup DDP if using multiple GPUs
        if world_size > 1:
            setup_ddp(rank, world_size)
        
        # Initialize components
        self.model = self._setup_model()
        self.criterion = self._setup_criterion()
        self.wandb = self._setup_wandb()
    
    def _setup_model(self) -> nn.Module:
        """Setup and initialize the model"""
        model = DenoisingNet(
            in_channels=self.config['model']['in_channels'],
            out_channels=self.config['model']['out_channels']
        ).to(self.device)
        
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.rank])
        
        return model
    
    def _setup_criterion(self) -> nn.Module:
        """Setup the loss function as a combination of L1 and TV loss"""
        main_loss = L1Loss()
        tv_loss = TVLoss()
        tv_weight = self.config['loss']['tv_weight']
        class CombinedLoss(nn.Module):
            def __init__(self, main_loss, tv_loss, tv_weight):
                super().__init__()
                self.main_loss = main_loss
                self.tv_loss = tv_loss
                self.tv_weight = tv_weight
            def forward(self, output, target):
                return self.main_loss(output, target) + self.tv_weight * self.tv_loss(output)
        return CombinedLoss(main_loss, tv_loss, tv_weight)
    
    def _setup_wandb(self) -> Optional[WandbWrapper]:
        """Setup Weights & Biases logging"""
        if 'wandb' in self.config and self.rank == 0:
            wandb_wrapper = WandbWrapper()
            wandb_wrapper.init(self.config)
            return wandb_wrapper
        return None
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.world_size > 1:
            torch.distributed.destroy_process_group()

class TrainingSetup(BaseSetup):
    """Setup for training mode"""
    def __init__(self, config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        super().__init__(config, rank, world_size)
        self.train_loader, self.val_loader = self._setup_data_loaders()
        self.optimizer = self._setup_optimizer()
    
    def _setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Setup training and validation data loaders"""
        train_dataset = create_train_dataset(
            self.config['data']['train_dir'],
            self.config['data']['intervals']
        )
        val_dataset = create_val_dataset(
            self.config['data']['val_dir']['noise'],
            self.config['data']['val_dir']['refer']
        )
        
        train_sampler = None
        val_sampler = None
        if self.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=self.world_size, rank=self.rank
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=self.world_size, rank=self.rank
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=(train_sampler is None),
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            sampler=train_sampler
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            sampler=val_sampler
        )
        
        return train_loader, val_loader
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup the optimizer"""
        return Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )

class ValidationSetup(BaseSetup):
    """Setup for validation/testing mode"""
    def __init__(self, config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        super().__init__(config, rank, world_size)
        self.val_loader = self._setup_data_loader()
    
    def _setup_data_loader(self) -> DataLoader:
        """Setup validation data loader"""
        val_dataset = create_val_dataset(
            self.config['data']['val_dir']['noise'],
            self.config['data']['val_dir']['refer']
        )
        
        val_sampler = None
        if self.world_size > 1:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=self.world_size, rank=self.rank
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            sampler=val_sampler
        )
        
        return val_loader 