
import os, sys
import yaml
from pathlib import Path
here = Path(__file__).parent

class YamlConfig:
    def __init__(self, config_dict, use_upper=False):
        for key, value in config_dict.items():
            if use_upper:
                key = key.upper()
            setattr(self, key, value)
            

def load_configs(config_path=None, include_env=False):
    """
    Load the configuration from a YAML file and environment variables.

    :param config_path: The path to the YAML config file. Defaults to "./config.yaml".
    :return: Merged configuration from environment variables and YAML file.
    """
    # Copy environment variables to avoid modifying them directly
    config_path = config_path or os.path.join(here, "config.yaml")
    configs = dict(os.environ) if include_env else {}
    if config_path and not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    try:
        with open(config_path, "r") as file:
            yaml_data = yaml.safe_load(file)
        # Update configs with YAML data
        if yaml_data:
            configs.update(yaml_data)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}. Using only environment variables.")
    return YamlConfig(configs, use_upper=True)
    # return configs

if __name__ == "__main__":
    cfg = load_configs()
    print(cfg)