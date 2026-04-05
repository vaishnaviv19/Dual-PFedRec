import torch
import torch.nn as nn


class GlobalItemEmbedding(nn.Module):
    """Global item embedding module E(θ_m) shared by all clients."""

    def __init__(self, num_items: int, embedding_dim: int):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return global item embeddings for the provided item ids."""
        return self.item_embedding(item_ids)

    def get_weights(self) -> torch.Tensor:
        """Return full global item embedding matrix for broadcast."""
        return self.item_embedding.weight.data.clone().cpu()

    def load_weights(self, weights: torch.Tensor):
        """Load aggregated global item embedding matrix θ_m."""
        self.item_embedding.weight.data = weights.clone().to(self.item_embedding.weight.device)

        