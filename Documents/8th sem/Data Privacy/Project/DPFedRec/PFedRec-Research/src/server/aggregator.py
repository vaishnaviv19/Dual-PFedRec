import torch
import torch.nn as nn
from typing import List, Dict, Any

class ServerAggregator:
    def __init__(self, num_items: int, embedding_size: int):
        self.num_items = num_items
        self.embedding_size = embedding_size
        
        # Initialize the global item embedding model
        self.global_model = nn.Embedding(num_items, embedding_size)
        nn.init.normal_(self.global_model.weight, std=0.01)
        
    def get_global_embedding_state(self) -> Dict[str, Any]:
        """Return the state_dict of the global model."""
        return self.global_model.state_dict()
        
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        FedAvg algorithm.
        client_updates: A list of state_dicts from clients.
        Each state_dict has key 'weight' which is the fine-tuned item parameter tensor.
        """
        if not client_updates:
            return self.get_global_embedding_state()
            
        avg_state_dict = {}
        for key in client_updates[0].keys():
            # Initialize sum tensor
            sum_tensor = torch.zeros_like(client_updates[0][key])
            
            # Accumulate all client tensors
            for sd in client_updates:
                sum_tensor += sd[key]
                
            # Average
            avg_state_dict[key] = sum_tensor / len(client_updates)
            
        # Update server
        self.global_model.load_state_dict(avg_state_dict)
        return self.get_global_embedding_state()
