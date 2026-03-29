# data/splitter.py
"""
Data Splitting Utilities
Implements leave-one-out evaluation (Paper Section 6.1)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional  # ✅ Added Optional import


def leave_one_out_split(interactions: np.ndarray, 
                       test_ratio: float = 0.2,
                       random_state: int = 42) -> Tuple[List[int], List[int]]:
    """
    Leave-one-out split for evaluation (Paper Section 6.1)
    
    For each user: last interaction = test, rest = train
    
    Args:
        interactions: Array of item IDs the user interacted with
        test_ratio: Fraction of interactions to use for testing
                   (for leave-one-out, this is typically just 1 item)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_items, test_items) lists
    """
    if len(interactions) == 0:
        return [], []
    
    # Set random state for reproducibility
    rng = np.random.RandomState(random_state)
    
    # For leave-one-out: take last interaction as test
    # Sort by timestamp if available, otherwise random
    shuffled = rng.permutation(interactions.copy())
    
    # Determine split point
    n_test = max(1, int(len(shuffled) * test_ratio))
    split_idx = len(shuffled) - n_test
    
    # Split
    train_items = shuffled[:split_idx].tolist()
    test_items = shuffled[split_idx:].tolist()
    
    return train_items, test_items


def split_dataset_by_users(df: pd.DataFrame, 
                          n_clients: Optional[int] = None,  # ✅ Now works
                          min_interactions: int = 20) -> Dict[int, pd.DataFrame]:
    """
    Split dataset into client-specific DataFrames (one user = one client)
    
    Args:
        df: Full interaction DataFrame
        n_clients: Number of clients to create (None = all users)
        min_interactions: Filter users with fewer interactions
        
    Returns:
        Dict mapping client_id to user-specific DataFrame
    """
    # Filter users by interaction count
    if min_interactions > 0:
        user_counts = df.groupby('user_id').size()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df['user_id'].isin(valid_users)].copy()
    
    # Determine which users to include
    unique_users = df['user_id'].unique()
    if n_clients is not None and n_clients < len(unique_users):
        # Sample n_clients users randomly
        rng = np.random.RandomState(42)
        selected_users = rng.choice(unique_users, size=n_clients, replace=False)
        unique_users = selected_users
    
    # Create client-specific DataFrames
    client_data = {}
    for i, user_id in enumerate(sorted(unique_users)):
        client_df = df[df['user_id'] == user_id].copy()
        client_data[i + 1] = client_df  # Client IDs start from 1
    
    return client_data


def create_non_iid_splits(df: pd.DataFrame,
                         n_clients: int,
                         alpha: float = 0.5,
                         random_state: int = 42) -> Dict[int, pd.DataFrame]:
    """
    Create non-IID data splits using Dirichlet distribution
    
    More realistic federated setting where users have different item preferences
    
    Args:
        df: Full interaction DataFrame
        n_clients: Number of clients to create
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        random_state: Random seed
        
    Returns:
        Dict mapping client_id to client-specific DataFrame
    """
    rng = np.random.RandomState(random_state)
    
    # Group interactions by item
    item_users = df.groupby('item_id')['user_id'].apply(list).to_dict()
    
    # Initialize client data
    client_data = {i: [] for i in range(n_clients)}
    
    # Assign each item's interactions to clients using Dirichlet
    for item_id, users in item_users.items():
        if len(users) == 0:
            continue
            
        # Sample client proportions for this item
        proportions = rng.dirichlet([alpha] * n_clients)
        
        # Assign each user interaction to a client
        for user_id in users:
            client_id = rng.choice(n_clients, p=proportions)
            client_data[client_id].append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': 1.0  # Implicit feedback
            })
    
    # Convert to DataFrames
    result = {}
    for client_id, interactions in client_data.items():
        if interactions:
            result[client_id] = pd.DataFrame(interactions)
    
    return result