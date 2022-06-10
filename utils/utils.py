import yaml

def load_cfg(cfg_path):
    with open(cfg_path, 'r') as stream:
        return yaml.safe_load(stream)