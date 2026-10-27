"""Module for handling configuration files."""

import json

class ConfigFile:
    """A class to handle configuration files."""

    def __init__(self, config_path):
        self.config_path = config_path
        self.config_data = self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading the configuration file: {e}")
            return None

    def get_config(self):
        return self.config_data