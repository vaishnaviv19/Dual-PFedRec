import torch
import torch.nn as nn
from typing import List

class PFedRec(nn.Module):
    def __init__(self, num_items: int, embedding_size: int, client_layers: List[int]):
        super(PFedRec, self).__init__()
        
        self.num_items = num_items
        self.embedding_size = embedding_size
        
        # Global Item Embeddings - \theta_m
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # Private Score Function - \theta_s (MLP)
        # Input: Item embedding. Output: score \in [0, 1]
        layers = []
        input_dim = embedding_size
        for hidden_dim in client_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        # Last layer outputs a single scalar (logit) to be passed through Sigmoid
        # Notice we assume `client_layers` last element is 1, e.g., [64, 32, 16, 1].
        # If the last element is 1, we don't need another Linear. Instead of manually
        # checking, we just append a sequence of standard Linear+ReLU and manually 
        # add the final layer if the provided ones are just hidden dimensions.
        # But our config says [64, 32, 16, 1]. The paper says to use Sigmoid.
        # Let's cleanly build it based on the list.
        # Since BCEWithLogitsLoss is more stable, we'll output logits, EXCEPT
        # the design: we will output logits and use BCEWithLogitsLoss in trainer.
        
        mlp_layers = []
        dim = embedding_size
        for h in client_layers[:-1]:
            mlp_layers.append(nn.Linear(dim, h))
            mlp_layers.append(nn.ReLU())
            dim = h
        
        # Final layer to 1 dim
        mlp_layers.append(nn.Linear(dim, 1))
        
        self.score_function = nn.Sequential(*mlp_layers)
        
    def forward(self, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Calculates the score for a batch of items.
        Returns the logits.
        """
        # (batch_size, embedding_size)
        item_embeds = self.item_embedding(item_indices)
        
        # (batch_size, 1) -> (batch_size,)
        logits = self.score_function(item_embeds).squeeze(-1)
        return logits

    def freeze_item_embeddings(self):
        """Step 1: Fix item embeddings, update score function."""
        for param in self.item_embedding.parameters():
            param.requires_grad = False
        for param in self.score_function.parameters():
            param.requires_grad = True

    def freeze_score_function(self):
        """Step 2: Fix score function, update fine-tuned item embeddings."""
        for param in self.item_embedding.parameters():
            param.requires_grad = True
        for param in self.score_function.parameters():
            param.requires_grad = False

    def load_global_embeddings(self, state_dict):
        """Load just the item_embedding from the server aggregated \theta_m"""
        self.item_embedding.load_state_dict(state_dict)

    def get_local_embeddings(self):
        """Return the fine-tuned item embeddings to send to server"""
        return self.item_embedding.state_dict()
