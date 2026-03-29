# data/sampler.py
"""Negative Sampling Module.

Implements Eq. 8: I⁻ᵢ = I \\ Iᵢ (uninteracted items)
"""

import numpy as np
from typing import List, Set, Union, Optional, Tuple


class NegativeSampler:
    """
    Negative sampler for implicit feedback recommendation
    
    Samples negative items from I⁻ᵢ = I \\ Iᵢ (Eq. 8)
    """
    
    def __init__(self, 
                 num_items: int,
                 ratio: int = 4,
                 seed: Optional[int] = None):
        """
        Args:
            num_items: Total number of items in catalog (|I|)
            ratio: Number of negatives to sample per positive
            seed: Random seed for reproducibility
        """
        self.num_items = num_items
        self.ratio = ratio
        self.rng = np.random.RandomState(seed)
        self.all_items = set(range(num_items))
    
    def sample(self, 
              positive_items: Union[np.ndarray, List[int]],
              negative_pool: Optional[Set[int]] = None,
              ratio: Optional[int] = None) -> np.ndarray:
        """
        Sample negative items for a batch of positives
        
        Args:
            positive_items: Array of positive item IDs
            negative_pool: Pre-computed I⁻ᵢ = I \\ Iᵢ (optional, for efficiency)
            ratio: Override default negative sampling ratio
            
        Returns:
            Array of sampled negative item IDs
        """
        if ratio is None:
            ratio = self.ratio
        
        # Convert to set for fast lookup
        if isinstance(positive_items, np.ndarray):
            positive_set = set(positive_items.tolist())
        else:
            positive_set = set(positive_items)
        
        # Compute negative pool if not provided (Eq. 8)
        if negative_pool is None:
            negative_pool = self.all_items - positive_set
        
        negative_pool = list(negative_pool)
        
        if len(negative_pool) == 0:
            # Edge case: user interacted with all items
            # Return empty array or repeat positives (shouldn't happen in practice)
            return np.array([], dtype=int)
        
        # Calculate number of negatives to sample
        n_pos = len(positive_items)
        n_neg = n_pos * ratio
        
        # Sample with replacement if pool is small
        replace = len(negative_pool) < n_neg
        
        negatives = self.rng.choice(
            negative_pool, 
            size=n_neg, 
            replace=replace
        )
        
        return np.array(negatives, dtype=int)
    
    def sample_for_user(self,
                       user_positive_items: Set[int],
                       n_samples: int) -> np.ndarray:
        """
        Sample negatives specifically for a user's positive items
        
        Args:
            user_positive_items: Set of items user has interacted with (Iᵢ)
            n_samples: Number of negative samples to return
            
        Returns:
            Array of negative item IDs
        """
        # Eq. 8: I⁻ᵢ = I \ Iᵢ
        negative_pool = self.all_items - user_positive_items
        
        if len(negative_pool) == 0:
            return np.array([], dtype=int)
        
        replace = len(negative_pool) < n_samples
        negatives = self.rng.choice(
            list(negative_pool),
            size=n_samples,
            replace=replace
        )
        
        return np.array(negatives, dtype=int)
    
    def create_training_batch(self,
                            positive_items: np.ndarray,
                            labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a training batch with positives and sampled negatives
        
        Args:
            positive_items: Array of positive item IDs
            labels: Optional labels array (default: 1 for pos, 0 for neg)
            
        Returns:
            Tuple of (item_ids, labels) for training
        """
        n_pos = len(positive_items)
        
        # Sample negatives
        negatives = self.sample(positive_items)
        
        # Combine items
        all_items = np.concatenate([positive_items, negatives])
        
        # Create labels
        if labels is None:
            pos_labels = np.ones(n_pos, dtype=np.float32)
            neg_labels = np.zeros(len(negatives), dtype=np.float32)
            all_labels = np.concatenate([pos_labels, neg_labels])
        else:
            all_labels = labels
        
        return all_items, all_labels