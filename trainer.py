from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, optimizer, criterion, device, train_dataset, val_dataset,training_params):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_dataloader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False)

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            self.model.train()
        
    def train(self):
        for batch in self.train_dataloader:
            self.optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
    def evaluate(self):