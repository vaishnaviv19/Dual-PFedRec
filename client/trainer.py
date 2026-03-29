# client/trainer.py
"""
PFedRec Client Training - Implements Algorithm 1 (ClientUpdate)
Dual Personalization: Update θₛ first, then fine-tune θₘ
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from .model import PFedRecModel
from data.sampler import NegativeSampler
from utils.logger import get_logger

logger = get_logger(__name__)


class PFedRecTrainer:
    """
    Client-side trainer implementing Algorithm 1: ClientUpdate
    """
    
    def __init__(self, model: PFedRecModel, config: Dict, 
                 negative_sampler: NegativeSampler):
        self.model = model
        self.config = config
        self.negative_sampler = negative_sampler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Loss function (Eq. 7)
        self.criterion = nn.BCELoss()
    
    def train_local(self, positive_items: np.ndarray, 
                   all_items: set) -> Tuple[torch.Tensor, Dict]:
        """
        Algorithm 1: ClientUpdate(i, θₘ)
        
        Args:
            positive_items: Items user i has interacted with
            all_items: Full item set I for negative sampling
            
        Returns:
            updated_item_embedding: Fine-tuned θₘᵢ to send to server
            metrics: Training metrics for logging
        """
        self.model.train()
        metrics = {"losses": [], "samples": 0}
        
        # Eq. 8: I⁻ᵢ = I \ Iᵢ (uninteracted items)
        negative_pool = list(all_items - set(positive_items))
        
        # Prepare positive items tensor
        pos_items_tensor = torch.tensor(positive_items, dtype=torch.long).to(self.device)
        
        # Local training loop (Algorithm 1, Lines 6-11)
        for epoch in range(self.config["epochs_local"]):
            # Shuffle positive items
            indices = torch.randperm(len(pos_items_tensor))
            pos_items_shuffled = pos_items_tensor[indices]
            
            # Mini-batch training
            for batch_start in range(0, len(pos_items_shuffled), 
                                   self.config["batch_size"]):
                batch_end = min(batch_start + self.config["batch_size"], 
                              len(pos_items_shuffled))
                batch_pos = pos_items_shuffled[batch_start:batch_end]
                
                # Sample negatives (Eq. 8)
                batch_neg = self.negative_sampler.sample(
                    positive_items=batch_pos.cpu().numpy(),
                    negative_pool=negative_pool,
                    ratio=self.config["negative_sampling_ratio"]
                )
                batch_neg = torch.tensor(batch_neg, dtype=torch.long).to(self.device)
                
                # === Step 1: Update Score Function θₛ (keeping θₘ fixed) ===
                # Algorithm 1, Lines 7-9
                self.model.set_requires_grad(score_fn=True, item_emb=False)
                
                # Forward pass for positives
                pos_pred = self.model(batch_pos)
                pos_loss = self.criterion(pos_pred, torch.ones_like(pos_pred))
                
                # Forward pass for negatives
                neg_pred = self.model(batch_neg)
                neg_loss = self.criterion(neg_pred, torch.zeros_like(neg_pred))
                
                # Total loss (Eq. 7)
                loss = pos_loss + neg_loss
                
                # Backward and update θₛ only
                self.model.optimizer_score.zero_grad()
                loss.backward()
                self.model.optimizer_score.step()
                
                metrics["losses"].append(loss.item())
                metrics["samples"] += len(batch_pos) + len(batch_neg)
                
                # === Step 2: Fine-tune Item Embedding θₘ (keeping θₛ fixed) ===
                # Algorithm 1, Lines 10-11
                self.model.set_requires_grad(score_fn=False, item_emb=True)
                
                # Re-compute loss with updated score function
                pos_pred = self.model(batch_pos)
                neg_pred = self.model(batch_neg)
                loss = self.criterion(pos_pred, torch.ones_like(pos_pred)) + \
                       self.criterion(neg_pred, torch.zeros_like(neg_pred))
                
                # Backward and update θₘ only
                self.model.optimizer_item.zero_grad()
                loss.backward()
                self.model.optimizer_item.step()
                
                metrics["losses"].append(loss.item())
        
        # Return fine-tuned item embedding (Algorithm 1, Line 12)
        updated_embedding = self.model.get_item_embedding_weights().cpu()
        
        # Compute average loss for logging
        avg_loss = np.mean(metrics["losses"]) if metrics["losses"] else 0
        
        logger.info(f"Client training complete: avg_loss={avg_loss:.4f}, "
                   f"samples={metrics['samples']}")
        
        return updated_embedding, {"avg_loss": avg_loss, "samples": metrics["samples"]}
    
    # Add this method to PFedRecTrainer class:

    def evaluate(self, test_items: np.ndarray, train_items: set, 
                all_items: set, k: int = 10) -> Dict[str, float]:
        """
        Evaluate model on test items (leave-one-out)
        """
        if len(test_items) == 0:
            return {"hr@10": 0.0, "ndcg@10": 0.0}
        
        self.model.eval()
        test_item = test_items[0]
        
        # Candidate set
        candidates = [test_item]
        negative_pool = list(all_items - train_items - {test_item})
        if len(negative_pool) >= 99:
            candidates.extend(np.random.choice(negative_pool, size=99, replace=False))
        else:
            candidates.extend(negative_pool)
        
        # Predict
        candidate_tensor = torch.tensor(candidates, dtype=torch.long).to(self.device)
        with torch.no_grad():
            scores = self.model(candidate_tensor).cpu().numpy()
        
        # Rank
        ranked = np.array(candidates)[np.argsort(-scores)]
        
        # Metrics
        hr = hit_ratio(ranked, [test_item], k)
        ndcg_score = ndcg(ranked, [test_item], k)
        
        return {"hr@10": float(hr), "ndcg@10": float(ndcg_score)}