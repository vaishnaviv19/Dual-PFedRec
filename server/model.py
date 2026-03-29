# client/model.py
"""PFedRec Model Architecture - Dual Personalization"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional

class PFedRecModel(nn.Module):
    """
    PFedRec Model (Paper Section 4.1)
    - Item Embedding Module (E): Shared globally, fine-tuned locally
    - Score Function (S): Personalized, never leaves device
    """
    
    def __init__(self, num_items: int, embedding_dim: int, 
                 score_hidden_dims: List[int] = [64, 32]):
        super(PFedRecModel, self).__init__()
        
        # Item Embedding Module (theta_m)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Score Function (theta_s) - Personalized MLP
        layers = []
        input_dim = embedding_dim
        for hidden_dim in score_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.score_function = nn.Sequential(*layers)
        
        # Separate optimizers for dual personalization
        self.optimizer_score = optim.SGD(
            self.score_function.parameters(),
            lr=0.01  # Will be set from config
        )
        self.optimizer_item = optim.SGD(
            self.item_embedding.parameters(),
            lr=0.001  # Will be set from config
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Eq. 6: r_hat = S(E(e_j))"""
        emb = self.item_embedding(item_ids)
        score = self.score_function(emb)
        return score.squeeze(-1)
    
    def get_item_embedding_weights(self) -> torch.Tensor:
        """Return item embedding for federated aggregation"""
        return self.item_embedding.weight.data.clone().cpu()
    
    def load_item_embedding_weights(self, weights: torch.Tensor):
        """Load global item embedding from server"""
        self.item_embedding.weight.data = weights.clone().to(self.device)
    
    def set_requires_grad(self, score_fn: bool, item_emb: bool):
        """Control which parameters require gradients for dual personalization"""
        for param in self.score_function.parameters():
            param.requires_grad = score_fn
        for param in self.item_embedding.parameters():
            param.requires_grad = item_emb