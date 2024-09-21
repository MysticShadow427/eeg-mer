from torch.utils.data import DataLoader
from custom_losses import ClassificationLoss, RegressionLoss, SupConLoss
import torch

class Trainer:
    def __init__(self, model, device, train_dataset, val_dataset,training_params):
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=training_params['learning_rate'])
        self.cls_criterion = ClassificationLoss() # music emotion classification loss
        self.reg_criterion = RegressionLoss() # arousal prediction loss
        self.con_criterion = SupConLoss() # contrastive loss
        self.device = device
        self.train_dataloader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False)

    def fit(self):
        for epoch in range(epochs):
            self.train()
            self.validate()
        
    def train(self):
        for batch in self.train_dataloader:
            
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                
                
    def evaluate(self):