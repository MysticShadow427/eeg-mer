import yaml

with open('config_ct.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

print(cfg)

training_params = cfg['training_params']