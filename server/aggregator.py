# server/aggregator.py
"""
Server-side Aggregation Logic - Algorithm 1 (ServerExecute)
Implements FedAvg for item embeddings (θₘ)
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from utils.logger import get_logger

logger = get_logger(__name__)


class FederatedAggregator:
    """
    Aggregates item embeddings from clients using FedAvg
    (Algorithm 1, Line 6: θₘ ← (1/n) Σ θₘᵢ)
    """
    
    def __init__(self, num_items: int, embedding_dim: int, 
                 aggregation_method: str = "fedavg"):
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.aggregation_method = aggregation_method
        
        # Initialize global item embedding randomly
        self.global_embedding = torch.randn(num_items, embedding_dim) * 0.01
        
        # Track client updates
        self.client_updates: Dict[str, Dict] = {}
        self.current_round = 0
        
        logger.info(f"Aggregator initialized: {num_items} items, "
                   f"{embedding_dim}d embeddings")
    
    def reset(self):
        """Reset aggregator state for new experiment"""
        self.global_embedding = torch.randn(
            self.num_items, self.embedding_dim) * 0.01
        self.client_updates.clear()
        self.current_round = 0
    
    def receive_update(self, client_id: str, embedding: torch.Tensor, 
                      num_samples: int, round_num: int) -> bool:
        """
        Receive and store client update
        
        Args:
            client_id: Unique client identifier
            embedding: Fine-tuned item embedding θₘᵢ (num_items × embedding_dim)
            num_samples: Number of training samples used by client
            round_num: Current federated round
            
        Returns:
            success: Whether update was accepted
        """
        if round_num != self.current_round:
            logger.warning(f"Stale update from client {client_id}: "
                          f"round {round_num} vs current {self.current_round}")
            return False
        
        # Validate embedding shape
        if embedding.shape != (self.num_items, self.embedding_dim):
            logger.error(f"Invalid embedding shape from client {client_id}: "
                        f"{embedding.shape} vs expected "
                        f"({self.num_items}, {self.embedding_dim})")
            return False
        
        # Store update
        self.client_updates[client_id] = {
            "embedding": embedding,
            "samples": num_samples,
            "timestamp": torch.cuda.current_device() if torch.cuda.is_available() else -1
        }
        
        logger.info(f"Received update from client {client_id} "
                   f"(round {round_num}, {num_samples} samples)")
        return True
    
    def aggregate(self, min_clients: int = 2) -> Optional[torch.Tensor]:
        """
        Perform FedAvg aggregation (Algorithm 1, Line 6)
        
        θₘ ← (1/n) Σᵢ₌₁ⁿ θₘᵢ   [weighted by sample count]
        
        Args:
            min_clients: Minimum clients required before aggregating
            
        Returns:
            new_global_embedding: Aggregated embedding, or None if not enough clients
        """
        if len(self.client_updates) < min_clients:
            logger.info(f"Waiting for more clients: "
                       f"{len(self.client_updates)}/{min_clients}")
            return None
        
        if self.aggregation_method == "fedavg":
            new_embedding = self._fedavg_aggregate()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        # Update global state
        self.global_embedding = new_embedding
        self.current_round += 1
        self.client_updates.clear()
        
        logger.info(f"✓ Aggregation complete: Round {self.current_round}, "
                   f"{len(self.client_updates) + min_clients} clients aggregated")
        
        return new_embedding
    
    def _fedavg_aggregate(self) -> torch.Tensor:
        """
        FedAvg: Weighted average based on client sample counts
        """
        total_samples = sum(u["samples"] for u in self.client_updates.values())
        
        if total_samples == 0:
            # Fallback to simple average
            embeddings = [u["embedding"] for u in self.client_updates.values()]
            return torch.stack(embeddings).mean(dim=0)
        
        # Weighted average
        aggregated = torch.zeros_like(self.global_embedding)
        for update in self.client_updates.values():
            weight = update["samples"] / total_samples
            aggregated += update["embedding"] * weight
        
        return aggregated
    
    def get_global_embedding(self) -> torch.Tensor:
        """Return current global item embedding for client download"""
        return self.global_embedding.clone()
    
    def get_stats(self) -> Dict:
        """Return aggregator statistics for monitoring"""
        return {
            "current_round": self.current_round,
            "pending_updates": len(self.client_updates),
            "embedding_shape": list(self.global_embedding.shape),
            "embedding_norm": self.global_embedding.norm().item()
        }