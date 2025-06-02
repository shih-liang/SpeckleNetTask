"""Weights & Biases wrapper for experiment tracking.

This module provides a wrapper around Weights & Biases functionality to make it
optional and handle cases where wandb is not installed.
"""

import os
from typing import Dict, Any, Optional, Union
import torch.nn as nn

class WandbWrapper:
    """Wrapper class for Weights & Biases functionality."""
    
    def __init__(self):
        """Initialize the wrapper and try to import wandb."""
        self.wandb = None
        self.initialized = False
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            print("Warning: wandb not installed. Running without experiment tracking.")
    
    def init(self, config: Dict[str, Any]) -> bool:
        """Initialize Weights & Biases.
        
        Args:
            config: Configuration dictionary containing wandb settings
            
        Returns:
            bool: True if wandb was successfully initialized, False otherwise
        """
        if self.wandb is None:
            return False
            
        try:
            self.wandb.init(
                project=config['wandb']['project'],
                config=config,
                name=config['wandb'].get('name', None),
                tags=config['wandb'].get('tags', []),
                notes=config['wandb'].get('notes', '')
            )
            self.initialized = True
            return True
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            return False
    
    def watch(self, model: nn.Module, log: str = "all") -> None:
        """Watch model parameters and gradients.
        
        Args:
            model: PyTorch model to watch
            log: What to log (default: "all")
        """
        if self.wandb is not None and self.initialized:
            try:
                self.wandb.watch(model, log=log)
            except Exception as e:
                print(f"Warning: Failed to watch model: {e}")
    
    def log(self, data: Dict[str, Any]) -> None:
        """Log metrics to wandb.
        
        Args:
            data: Dictionary of metrics to log
        """
        if self.wandb is not None and self.initialized:
            try:
                self.wandb.log(data)
            except Exception as e:
                print(f"Warning: Failed to log metrics: {e}")
    
    def save(self, path: Union[str, os.PathLike]) -> None:
        """Save a file to wandb.
        
        Args:
            path: Path to the file to save
        """
        if self.wandb is not None and self.initialized:
            try:
                self.wandb.save(str(path))
            except Exception as e:
                print(f"Warning: Failed to save file: {e}")
    
    def finish(self) -> None:
        """Finish the wandb run."""
        if self.wandb is not None and self.initialized:
            try:
                self.wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")

# Create a global instance
wandb_wrapper = WandbWrapper()