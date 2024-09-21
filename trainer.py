from torch.utils.data import DataLoader
from custom_losses import ClassificationLoss, RegressionLoss, SupConLoss, person_specific_loss
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from custom_dataset import collate_fn
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, device, train_dataset, val_dataset, training_params):
        self.model = model.to(device)
        self.training_params = training_params
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=training_params['learning_rate'])
        self.cls_criterion = ClassificationLoss()  # music emotion classification loss
        self.person_cls_criterion = ClassificationLoss()
        self.reg_criterion = RegressionLoss()  # arousal prediction loss
        self.con_criterion = SupConLoss()  # contrastive loss
        self.person_criterion = person_specific_loss()  # person-specific loss
        self.device = device
        self.train_dataloader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True,collate_fn=collate_fn)
        self.val_dataloader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False,collate_fn=collate_fn)
        self.train_losses = []
        self.val_losses = []

    def fit(self):
        for epoch in range(self.training_params['epochs']):
            print(f"Epoch {epoch+1}/{self.training_params['epochs']}")
            train_loss = self.train()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
        self.save_model()

    def train(self):
        self.model.train()
        running_loss = 0.0

        for batch in tqdm(self.train_dataloader, desc="Training"):
           
            inputs, targets = batch['eeg_signal'].to(self.device), batch['targets'].to(self.device)
            self.optimizer.zero_grad()

            class_logits, person_logits, x_e, x_q = self.model(inputs)

            cls_loss = self.cls_criterion(class_logits, targets['class_labels'])
            person_cls_loss = self.person_cls_criterion(person_logits, targets['person_labels'])
            reg_loss = self.reg_criterion(x_q, targets['arousal'])
            con_loss = self.con_criterion(x_e)
            
            person_loss = self.person_criterion(
                person_count=self.model.vqvae.person_count, 
                activation_counters=self.model.vqvae.activation_counters,
                num_embeddings=self.model.vqvae.num_embeddings
            )

            total_loss = cls_loss + person_cls_loss + reg_loss + con_loss + person_loss

            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()

        epoch_loss = running_loss / len(self.train_dataloader)
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                inputs, targets = batch['inputs'].to(self.device), batch['targets'].to(self.device)

                class_logits, person_logits, x_e, x_q = self.model(inputs)

                cls_loss = self.cls_criterion(class_logits, targets['class_labels'])
                person_cls_loss = self.person_cls_criterion(person_logits, targets['person_labels'])
                reg_loss = self.reg_criterion(x_q, targets['arousal'])
                con_loss = self.con_criterion(x_e)
                
                person_loss = self.person_criterion(
                    person_count=self.model.vqvae.person_count, 
                    activation_counters=self.model.vqvae.activation_counters,
                    num_embeddings=self.model.vqvae.num_embeddings
                )

                total_loss = cls_loss + person_cls_loss + reg_loss + con_loss + person_loss

                running_loss += total_loss.item()

        epoch_loss = running_loss / len(self.val_dataloader)
        return epoch_loss

    def evaluate(self):
        self.model.eval()
        
        train_true, train_pred = [], []
        val_true, val_pred = []

        with torch.no_grad():
            for batch in tqdm(self.train_dataloader, desc="Evaluating on Train Set"):
                inputs, targets = batch['inputs'].to(self.device), batch['targets'].to(self.device)
                class_logits, _, _, _ = self.model(inputs)

                pred_labels = torch.argmax(class_logits, dim=1).cpu().numpy()
                true_labels = targets['class_labels'].cpu().numpy()

                train_true.extend(true_labels)
                train_pred.extend(pred_labels)

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating on Val Set"):
                inputs, targets = batch['inputs'].to(self.device), batch['targets'].to(self.device)
                class_logits, _, _, _ = self.model(inputs)

                pred_labels = torch.argmax(class_logits, dim=1).cpu().numpy()
                true_labels = targets['class_labels'].cpu().numpy()

                val_true.extend(true_labels)
                val_pred.extend(pred_labels)

        # Calculate metrics for the training set
        train_accuracy = accuracy_score(train_true, train_pred)
        train_precision_micro = precision_score(train_true, train_pred, average='micro')
        train_precision_macro = precision_score(train_true, train_pred, average='macro')
        train_precision_weighted = precision_score(train_true, train_pred, average='weighted')
        train_recall_micro = recall_score(train_true, train_pred, average='micro')
        train_recall_macro = recall_score(train_true, train_pred, average='macro')
        train_recall_weighted = recall_score(train_true, train_pred, average='weighted')
        train_conf_matrix = confusion_matrix(train_true, train_pred)

        # Calculate metrics for the validation set
        val_accuracy = accuracy_score(val_true, val_pred)
        val_precision_micro = precision_score(val_true, val_pred, average='micro')
        val_precision_macro = precision_score(val_true, val_pred, average='macro')
        val_precision_weighted = precision_score(val_true, val_pred, average='weighted')
        val_recall_micro = recall_score(val_true, val_pred, average='micro')
        val_recall_macro = recall_score(val_true, val_pred, average='macro')
        val_recall_weighted = recall_score(val_true, val_pred, average='weighted')
        val_conf_matrix = confusion_matrix(val_true, val_pred)

        # Print metrics for the training set
        print("\nTraining Set Metrics:")
        print(f"Accuracy: {train_accuracy:.4f}")
        print(f"Precision (Micro): {train_precision_micro:.4f}, Precision (Macro): {train_precision_macro:.4f}, Precision (Weighted): {train_precision_weighted:.4f}")
        print(f"Recall (Micro): {train_recall_micro:.4f}, Recall (Macro): {train_recall_macro:.4f}, Recall (Weighted): {train_recall_weighted:.4f}")
        print("Confusion Matrix:")
        print(train_conf_matrix)

        # Print metrics for the validation set
        print("\nValidation Set Metrics:")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"Precision (Micro): {val_precision_micro:.4f}, Precision (Macro): {val_precision_macro:.4f}, Precision (Weighted): {val_precision_weighted:.4f}")
        print(f"Recall (Micro): {val_recall_micro:.4f}, Recall (Macro): {val_recall_macro:.4f}, Recall (Weighted): {val_recall_weighted:.4f}")
        print("Confusion Matrix:")
        print(val_conf_matrix)


    def save_model(self):
        model_save_path = '/content/eeg-mer/model.pth'
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    
        

    def plot_loss_curves(self):
        """
        Plots the training and validation loss curves.

        Args:
        - train_losses (list): List of training losses per epoch.
        - val_losses (list): List of validation losses per epoch.
        """
        plt.figure(figsize=(10, 6))

        plt.plot(self.train_losses, label="Training Loss", color='blue', marker='o')
        plt.plot(self.val_losses, label="Validation Loss", color='orange', marker='o')

        plt.title("Training and Validation Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        
        plt.legend()
        
        plt.grid(True)
        plt.show()
        plt.savefig('loss_curve.png')


