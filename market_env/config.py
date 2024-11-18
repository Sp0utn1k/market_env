"""
Configuration Management Module.

This module handles loading and accessing configuration settings.
"""

import json
import os


class Config:
    """Configuration Loader and Accessor."""

    def __init__(self, external_config_dir: str, config_name: str = 'default'):
        """
        Initializes the Config object by loading the specified configuration file.

        Args:
            config_name (str): Name of the configuration file without extension.
            external_config_dir (str): Optional directory to look for configurations outside the library.
        """
        self.config = self._load_config(config_name, external_config_dir)

    def _load_config(self, config_name: str, external_config_dir: str) -> dict:
        """Loads a JSON configuration file, checking an external directory first if provided."""
        try:
            # First, check if an external directory is provided and the file exists there
            if external_config_dir:
                external_path = os.path.abspath(os.path.join(external_config_dir, f'{config_name}.json'))
                with open(external_path, 'r') as file:
                    return json.load(file)

            # Fallback to loading from the library's own config directory
            package_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(package_dir, 'configs', f'{config_name}.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file '{config_name}.json' not found in configs directory.")
            with open(config_path, 'r') as file:
                return json.load(file)

        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON configuration file '{config_name}.json': {e}")
        except Exception as e:
            raise Exception(f"An error occurred while loading the configuration file '{config_name}.json': {e}")

    def get(self, key: str, default=None):
        """Retrieves a configuration value."""
        return self.config.get(key, default)

    def get(self, key: str, default=None):
        """Retrieves a configuration value."""
        return self.config.get(key, default)
