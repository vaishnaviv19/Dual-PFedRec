import torch
import numpy as np
from typing import Dict, Optional, Tuple
from server.config import ServerConfig

class FederatedAggregator:
    
    def __init__(self, num_items: int, embedding_dim: int, aggregation_method: str = "fedavg"):
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.aggregation_method = aggregation_method
        self.global_embedding: Optional[torch.Tensor] = None
        self.current_round: int = 0
        self.client_updates: Dict[str, Dict] = {}
        self.reset()
    
    def reset(self):
        """Initialize global item embedding"""
        self.global_embedding = torch.randn(self.num_items, self.embedding_dim)
        self.current_round = 0
        self.client_updates = {}
    
    def get_global_embedding(self) -> torch.Tensor:
        """Return current global item embedding"""
        if self.global_embedding is None:
            self.reset()
        return self.global_embedding.clone()
    
    def receive_update(self, client_id: str, embedding: torch.Tensor, num_samples: int, round_num: int) -> bool:
        
        # Validate embedding shape
        if embedding.shape != (self.num_items, self.embedding_dim):
            return False
        
        # Store update
        self.client_updates[client_id] = {
            'embedding': embedding.clone(),
            'samples': num_samples,
            'round': round_num
        }
        return True
    
    def aggregate(self, min_clients: int = 2) -> Optional[torch.Tensor]:
        if len(self.client_updates) < min_clients:
            return None
        
        # Calculate total samples for weighting
        total_samples = sum(u['samples'] for u in self.client_updates.values())
        
        # Weighted average 
        aggregated = torch.zeros(self.num_items, self.embedding_dim)
        for update in self.client_updates.values():
            weight = update['samples'] / total_samples
            aggregated += update['embedding'] * weight
        
        self.global_embedding = aggregated
        self.current_round += 1
        self.client_updates = {}
        
        print(f"Round {self.current_round} aggregation complete")
        return self.global_embedding.clone()
    
    def get_stats(self) -> Dict[str, any]:
        """Get aggregator statistics"""
        return {
            'current_round': self.current_round,
            'pending_updates': len(self.client_updates),
            'embedding_shape': list(self.global_embedding.shape) if self.global_embedding is not None else None
        }