"""
Configuration loader for the multi-agent failure attribution system
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
    
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_config(config: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get model configuration for specified model name.
    
    Args:
        config: Full configuration dictionary
        model_name: Name of model to use (if None, uses config['system']['model_name'])
    
    Returns:
        Model configuration dictionary
    """
    if model_name is None:
        model_name = config['system']['model_name']
    
    if model_name not in config['models']:
        raise ValueError(f"Model '{model_name}' not found in config. Available: {list(config['models'].keys())}")
    
    return config['models'][model_name]


def get_system_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get system configuration.
    
    Args:
        config: Full configuration dictionary
    
    Returns:
        System configuration dictionary
    """
    return config['system']


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get data configuration.
    
    Args:
        config: Full configuration dictionary
    
    Returns:
        Data configuration dictionary
    """
    return config.get('data', {})


def get_output_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get output configuration.
    
    Args:
        config: Full configuration dictionary
    
    Returns:
        Output configuration dictionary
    """
    return config.get('output', {})

