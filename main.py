import torch
import yaml
from trainer import Trainer
from custom_dataset import EEGDataset
from model import EEGModel
from utils import print_blue


print_blue("Loading configuration file...")
with open('config_ct.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

training_params = cfg['training_params']
print_blue("Configuration file loaded successfully.")

print_blue("Setting up device for training...")
device = torch.device('cuda' if torch.cuda.is_available() and training_params['device'] == 'cuda' else 'cpu')
print_blue(f"Using device: {device}")

print_blue("Loading train and validation datasets...")
train_dataset = EEGDataset(split='train') 
val_dataset = EEGDataset(split='val')
print_blue("Datasets loaded successfully.")

model_type = cfg['architecture']
model_cfg = cfg['model']
vqvae_cfg = cfg['vqvae']
print_blue(f"Initializing the {model_type} model...")

model = EEGModel(model=model_type, cfg=model_cfg, vqvae_cfg=vqvae_cfg)
print_blue(f"{model_type} model initialized successfully.")

print_blue("Initializing trainer with the model and datasets...")
trainer = Trainer(model=model, device=device, train_dataset=train_dataset, val_dataset=val_dataset, training_params=training_params)
print_blue("Trainer initialized successfully.")

print_blue("Starting training process...")
trainer.fit()
print_blue("Training process completed.")

print_blue("Starting evaluation on train and validation sets...")
trainer.evaluate()
print_blue("Evaluation completed.")
