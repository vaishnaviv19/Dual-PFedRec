import torch
import torch.nn as nn

class FedMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_size: int):
        super(FedMF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        
        # User and Item Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        
        # Initialization
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Matrix Factorization using dot product between user and item embeddings.
        Returns the logits (unscaled scores).
        """
        u_embeds = self.user_embedding(user_indices) # (batch_size, embedding_size)
        i_embeds = self.item_embedding(item_indices) # (batch_size, embedding_size)
        
        # Element-wise product and sum across dim 1
        return (u_embeds * i_embeds).sum(1)
        
    def load_global_embeddings(self, state_dict):
        """Load just the item_embedding from the server aggregated \theta_m"""
        self.item_embedding.load_state_dict(state_dict)

    def get_local_embeddings(self):
        """Return the fine-tuned item embeddings to send to server"""
        return self.item_embedding.state_dict()
