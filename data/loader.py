# data/loader.py
"""
MovieLens Data Loader
Loads and preprocesses interaction data for federated recommendation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional


def load_movielens_100k(data_dir: str = "data/ml-100k/") -> pd.DataFrame:
    """
    Load MovieLens-100K dataset
    
    Args:
        data_dir: Path to ml-100k directory containing u.data
        
    Returns:
        DataFrame with columns: user_id, item_id, rating, timestamp
    """
    data_path = Path(data_dir) / "u.data"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"MovieLens data not found at {data_path}. "
            "Download from: https://grouplens.org/datasets/movielens/100k/"
        )
    
    # Load tab-separated file (no header)
    df = pd.read_csv(
        data_path, 
        sep='\t', 
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        dtype={
            'user_id': int,
            'item_id': int, 
            'rating': float,
            'timestamp': int
        }
    )
    
    # Convert to 0-based indexing (PyTorch convention)
    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1
    
    # Keep only positive interactions (implicit feedback)
    # Paper uses implicit feedback: rating > 0 → interaction
    df = df[df['rating'] > 0].copy()
    
    return df


def filter_users_by_interactions(df: pd.DataFrame, 
                                  min_interactions: int = 20) -> pd.DataFrame:
    """
    Filter users with fewer than min_interactions
    
    Args:
        df: Interaction DataFrame
        min_interactions: Minimum interactions per user
        
    Returns:
        Filtered DataFrame
    """
    user_counts = df.groupby('user_id').size()
    valid_users = user_counts[user_counts >= min_interactions].index
    return df[df['user_id'].isin(valid_users)].copy()


def create_interaction_matrix(df: pd.DataFrame, 
                             num_users: int, 
                             num_items: int) -> np.ndarray:
    """
    Create sparse user-item interaction matrix
    
    Args:
        df: Interaction DataFrame
        num_users: Total number of users
        num_items: Total number of items
        
    Returns:
        Binary interaction matrix (num_users × num_items)
    """
    matrix = np.zeros((num_users, num_items), dtype=np.float32)
    
    for _, row in df.iterrows():
        matrix[int(row['user_id']), int(row['item_id'])] = 1.0
    
    return matrix


def get_user_interactions(df: pd.DataFrame, 
                         user_id: int) -> np.ndarray:
    """
    Get all items a user has interacted with
    
    Args:
        df: Interaction DataFrame
        user_id: User index (0-based)
        
    Returns:
        Array of item IDs the user interacted with
    """
    items = df[df['user_id'] == user_id]['item_id'].values
    return np.unique(items)


def load_client_data(file_path: str) -> Tuple[int, np.ndarray]:
    """
    Load data for a single client (user)
    
    Args:
        file_path: Path to client CSV file
        
    Returns:
        Tuple of (user_id, array of interacted item IDs)
    """
    df = pd.read_csv(file_path)
    
    # Verify single user
    users = df['user_id'].unique()
    if len(users) != 1:
        raise ValueError(f"Client file should contain exactly one user, found: {users}")
    
    user_id = int(users[0])
    items = df['item_id'].values.astype(int)
    
    return user_id, items