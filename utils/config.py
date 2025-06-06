"""Configuration manager for training.

This module handles loading and validating training configuration.
"""

import os
from typing import Dict, Any
import yaml

class Config:
    """Configuration manager for training"""
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._validate_config()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from YAML file"""
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
        
        return process_dict(config)
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        required_sections = ['data', 'model', 'training', 'loss']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section '{section}' in config")
        
        # Validate data section
        data_config = self.config['data']
        required_data = ['train_dir', 'val_dir', 'intervals']
        for param in required_data:
            if param not in data_config:
                raise ValueError(f"Missing required parameter '{param}' in data section")
        
        # Validate val_dir structure
        val_dir = data_config['val_dir']
        if not isinstance(val_dir, dict) or 'noise' not in val_dir or 'refer' not in val_dir:
            raise ValueError("val_dir must be a dictionary with 'noise' and 'refer' keys")
        
        # Validate training section
        training_config = self.config['training']
        required_training = ['device', 'batch_size', 'num_workers', 'learning_rate', 
                           'epochs', 'save_interval', 'checkpoint_dir']
        for param in required_training:
            if param not in training_config:
                raise ValueError(f"Missing required parameter '{param}' in training section")
        
        # Validate model section
        model_config = self.config['model']
        required_model = ['in_channels', 'out_channels']
        for param in required_model:
            if param not in model_config:
                raise ValueError(f"Missing required parameter '{param}' in model section")
        
        # Validate loss section
        loss_config = self.config['loss']
        required_loss = ['tv_weight']
        for param in required_loss:
            if param not in loss_config:
                raise ValueError(f"Missing required parameter '{param}' in loss section")
        
        # Validate wandb section if present
        if 'wandb' in self.config:
            wandb_config = self.config['wandb']
            required_wandb = ['project', 'log_interval']
            for param in required_wandb:
                if param not in wandb_config:
                    raise ValueError(f"Missing required parameter '{param}' in wandb section") 