import torch
import numpy as np
from config import Config

class FederatedAggregator:
    """Implements FedAvg aggregation for item embeddings (Algorithm 1, Line 6)"""
    
    def __init__(self, config):
        self.config = config
        self.global_embedding = None
        self.current_round = 0
        self.client_count = 0
        self.client_updates = {}
        self.reset()
    
    def reset(self):
        """Initialize global item embedding"""
        self.global_embedding = torch.rand(
            self.config.NUM_ITEMS, 
            self.config.EMBEDDING_SIZE
        )
        self.current_round = 0
        self.client_updates = {}
        self.client_count = 0
    
    def get_global_embedding(self):
        """Return current global item embedding"""
        return self.global_embedding.clone()
    
    def aggregate(self, client_id, client_embedding, num_samples):
        """
        Aggregate client updates using FedAvg
        Paper Section 4.3: Server aggregates only item embeddings (theta_m)
        """
        self.client_updates[client_id] = {
            'embedding': client_embedding,
            'samples': num_samples
        }
        self.client_count = len(self.client_updates)
        
        # Check if enough clients have reported
        if self.client_count >= self.config.MIN_CLIENTS_PER_ROUND:
            self._perform_aggregation()
    
    def _perform_aggregation(self):
        """Perform FedAvg aggregation (Algorithm 1, Line 6)"""
        total_samples = sum(u['samples'] for u in self.client_updates.values())
        
        # Weighted average based on client data size
        aggregated = torch.zeros_like(self.global_embedding)
        for update in self.client_updates.values():
            weight = update['samples'] / total_samples
            aggregated += update['embedding'] * weight
        
        self.global_embedding = aggregated
        self.current_round += 1
        self.client_updates = {}
        self.client_count = 0
        
        print(f"✓ Round {self.current_round} aggregation complete")