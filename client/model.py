# client/model.py
"""
PFedRec Model Architecture (Paper Section 4.1)
- Item Embedding Module E(θₘ): Shared globally, fine-tuned locally
- Score Function S(θₛ): Personalized MLP, NEVER leaves device
- User Embedding: Removed (Section 4.1 - score function has enough capacity)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional


class ScoreFunction(nn.Module):
    """
    Personalized Score Function S(θₛ) - Eq. 6: r̂ = S(E(eⱼ))
    Multi-layer MLP that stays PRIVATE on client device.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_activation: str = "sigmoid"):
        super(ScoreFunction, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_embeddings: (batch_size, embedding_dim)
        Returns:
            scores: (batch_size,) - predicted preference scores
        """
        return self.network(item_embeddings).squeeze(-1)


class PFedRecModel(nn.Module):
    """
    Complete PFedRec Model with Dual Personalization.
    
    Forward: r̂ᵢⱼ = Sᵢ(Eᵢ(eⱼ))  [Eq. 6]
    """
    def __init__(self, num_items: int, embedding_dim: int, 
                 score_hidden_dims: List[int]):
        super(PFedRecModel, self).__init__()
        
        # Item Embedding Module E(θₘ) - Federated component
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Personalized Score Function S(θₛ) - Private component
        self.score_function = ScoreFunction(
            input_dim=embedding_dim,
            hidden_dims=score_hidden_dims
        )
        
        # Separate optimizers for dual personalization (Alg. 1, Lines 9, 11)
        self._setup_optimizers()
    
    def _setup_optimizers(self, lr_score: float = 0.01, lr_item: float = 0.001):
        """Initialize separate optimizers for θₛ and θₘ"""
        self.optimizer_score = torch.optim.SGD(
            self.score_function.parameters(), lr=lr_score
        )
        self.optimizer_item = torch.optim.SGD(
            self.item_embedding.parameters(), lr=lr_item
        )
    
    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Eq. 6: r̂ᵢⱼ = Sᵢ(Eᵢ(eⱼ))
        
        Args:
            item_ids: (batch_size,) - item indices
        Returns:
            predictions: (batch_size,) - preference scores in [0, 1]
        """
        # Get item embeddings: (batch_size, embedding_dim)
        item_embs = self.item_embedding(item_ids)
        # Score function: (batch_size,)
        return self.score_function(item_embs)
    
    def compute_loss(self, item_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Binary Cross-Entropy Loss (Eq. 7)
        
        Lᵢ = -Σ log(r̂ᵢⱼ) - Σ log(1 - r̂ᵢⱼ')
        """
        predictions = self(item_ids)
        loss = nn.BCELoss()(predictions, labels.float())
        return loss
    
    def get_item_embedding_weights(self) -> torch.Tensor:
        """Return item embedding weights for federated aggregation"""
        return self.item_embedding.weight.data.clone()
    
    def load_item_embedding_weights(self, weights: torch.Tensor):
        """Load global item embedding weights from server"""
        self.item_embedding.weight.data = weights.clone()
    
    def get_score_function_state(self) -> Dict:
        """Return score function state dict (for local saving, NOT sharing)"""
        return self.score_function.state_dict()
    
    def set_requires_grad(self, score_fn: bool, item_emb: bool):
        """Control which parameters to update (Alg. 1, Lines 7-11)"""
        for param in self.score_function.parameters():
            param.requires_grad = score_fn
        for param in self.item_embedding.parameters():
            param.requires_grad = item_emb