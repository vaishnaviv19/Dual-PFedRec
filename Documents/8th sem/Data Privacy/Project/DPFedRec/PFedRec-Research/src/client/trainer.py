import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging

from src.models.pfedrec import PFedRec
from src.utils.metrics import evaluate_predictions

logger = logging.getLogger(__name__)

class UserDataset(Dataset):
    def __init__(self, data_list):
        # data_list is a list of (user_id, item_id, label)
        self.users = torch.tensor([u for u, _, _ in data_list], dtype=torch.long)
        self.items = torch.tensor([i for _, i, _ in data_list], dtype=torch.long)
        self.labels = torch.tensor([l for _, _, l in data_list], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

class ClientTrainer:
    def __init__(self, user_id, train_data, test_data, config, model):
        self.user_id = user_id
        
        self.train_dataset = UserDataset(train_data)
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config.get('batch_size', 256), 
            shuffle=True
        )
        
        self.test_positive = test_data['test_positive']
        self.test_negatives = test_data['test_negatives']
        
        self.config = config
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        
    def _train_epochs(self, num_epochs, lr, weight_decay=1e-4):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for users, items, labels in self.train_loader:
                optimizer.zero_grad()
                
                # Check which model is being used (PFedRec or FedMF)
                if isinstance(self.model, PFedRec):
                    preds = self.model(items)
                else:
                    preds = self.model(users, items)
                
                loss = self.criterion(preds, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(labels)
            
            # avg_loss = total_loss / len(self.train_dataset)
            # logger.debug(f"User {self.user_id} Epoch {epoch+1} Loss: {avg_loss:.4f}")

    def train_dual_personalization(self):
        """
        Executes Algorithm 1 from PFedRec paper.
        Step 1: Train score_function while freezing item embeddings
        Step 2: Train item_embeddings while freezing score_function
        """
        lr = self.config.get('lr', 0.001)
        w_d = self.config.get('weight_decay', 1e-4)

        if isinstance(self.model, PFedRec):
            # Step 1
            self.model.freeze_item_embeddings()
            self._train_epochs(self.config.get('local_epochs_step1', 2), lr, w_d)
            
            # Step 2
            self.model.freeze_score_function()
            self._train_epochs(self.config.get('local_epochs_step2', 2), lr, w_d)
        else:
            # For FedMF, just train naturally
            self._train_epochs(
                self.config.get('local_epochs_step1', 2) + self.config.get('local_epochs_step2', 2),
                lr, w_d
            )

    def evaluate(self):
        """
        Evaluate HR@10 and NDCG@10 on the test sets using the personalized model.
        """
        self.model.eval()
        with torch.no_grad():
            items = torch.tensor([self.test_positive] + self.test_negatives, dtype=torch.long)
            users = torch.tensor([self.user_id] * len(items), dtype=torch.long)
            
            if isinstance(self.model, PFedRec):
                preds = self.model(items)
            else:
                preds = self.model(users, items)
                
            # Logits can be compared directly for ranking
            preds = preds.numpy()
            
            # Ground truth labels: 1 for test_positive (first item), 0 for the rest
            labels = [1.0] + [0.0] * len(self.test_negatives)
            
            hr, ndcg = evaluate_predictions(preds, labels, k=10)
            return hr, ndcg
