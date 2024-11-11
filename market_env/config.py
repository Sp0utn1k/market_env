"""
Configuration Management Module.

This module handles loading and accessing configuration settings.
"""

import json
import os


class Config:
    """Configuration Loader and Accessor."""

    def __init__(self, config_name: str = 'default'):
        """
        Initializes the Config object by loading the specified configuration file.

        Args:
            config_name (str): Name of the configuration file without extension.
        """
        self.config = self._load_config(config_name)

    def _load_config(self, config_name: str) -> dict:
        """Loads a JSON configuration file."""
        package_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(package_dir, 'configs', f'{config_name}.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_name}.json not found in configs directory.")
        with open(config_path, 'r') as file:
            return json.load(file)

    def get(self, key: str, default=None):
        """Retrieves a configuration value."""
        return self.config.get(key, default)
