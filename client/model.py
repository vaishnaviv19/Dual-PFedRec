import torch
import torch.nn as nn
import torch.optim as optim
from config import Config

class PFedRecModel(nn.Module):
    """
    PFedRec Model Architecture (Paper Section 4.1)
    - Item Embedding Module (E): Shared globally, finetuned locally
    - Score Function (S): Personalized, never leaves device
    - User Embedding: Removed (Section 4.1)
    """
    
    def __init__(self, num_items, embed_size, hidden_size):
        super(PFedRecModel, self).__init__()
        
        # Item Embedding Module (theta_m)
        self.item_embedding = nn.Embedding(num_items, embed_size)
        
        # Score Function (theta_s) - Personalized MLP
        self.score_function = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Separate optimizers for dual personalization
        self.optimizer_score = optim.SGD(
            self.score_function.parameters(), 
            lr=Config.LEARNING_RATE_SCORE
        )
        self.optimizer_item = optim.SGD(
            self.item_embedding.parameters(), 
            lr=Config.LEARNING_RATE_ITEM
        )
    
    def forward(self, item_ids):
        """Eq. 6: r_hat = S(E(e_j))"""
        emb = self.item_embedding(item_ids)
        score = self.score_function(emb)
        return score.squeeze()
    
    def get_item_embedding(self):
        """Return item embedding for federated aggregation"""
        return self.item_embedding.weight.data.clone()
    
    def load_item_embedding(self, weights):
        """Load global item embedding from server"""
        self.item_embedding.weight.data = weights.clone()
        