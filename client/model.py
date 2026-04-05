import torch
import torch.nn as nn
from typing import List, Dict, Optional


class ScoreFunction(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_activation: str = "sigmoid"):
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
        return self.network(item_embeddings).squeeze(-1)


class PFedRecModel(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int, score_hidden_dims: Optional[List[int]] = None):
        super(PFedRecModel, self).__init__()

        if score_hidden_dims is None:
            # one-layer MLP for score function
            score_hidden_dims = []
        
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.score_function = ScoreFunction(
            input_dim=embedding_dim,
            hidden_dims=score_hidden_dims
        )
        
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
        
        item_embs = self.item_embedding(item_ids)
        
        return self.score_function(item_embs)
    
    def compute_loss(self, item_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        predictions = self(item_ids)
        loss = nn.BCELoss()(predictions, labels.float())
        return loss
    
    def get_item_embedding_weights(self) -> torch.Tensor:
        """Return θ_m """
        return self.item_embedding.weight.data.clone()
    
    def load_item_embedding_weights(self, weights: torch.Tensor):
        self.item_embedding.weight.data = weights.clone().to(self.item_embedding.weight.device)
    
    def get_score_function_state(self) -> Dict:
        """Return private θ_s state (local checkpointing only)"""
        return self.score_function.state_dict()
    
    def set_requires_grad(self, score_fn: bool, item_emb: bool):
        """Control which parameters to update """
        for param in self.score_function.parameters():
            param.requires_grad = score_fn
        for param in self.item_embedding.parameters():
            param.requires_grad = item_emb