import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

import logging
logger = logging.getLogger(__name__)

def generate_negative_samples(user_interacted: set, total_items: int, num_samples: int) -> List[int]:
    """Uniformly sample `num_samples` items not in `user_interacted`."""
    negatives = set()
    while len(negatives) < num_samples:
        random_item = np.random.randint(0, total_items)
        if random_item not in user_interacted and random_item not in negatives:
            negatives.add(random_item)
    return list(negatives)

def preprocess_user_data(
    user_df: pd.DataFrame, 
    total_items: int, 
    train_negatives: int = 4, 
    test_negatives: int = 99
) -> Dict:
    """
    Process a single user's dataframe.
    - Leave-one-out strategy: The last interacted item is the test target.
    - For each training positive item, sample `train_negatives` negative items.
    - For the test positive item, sample `test_negatives` negative items.
    """
    # Sort by time
    user_df = user_df.sort_values(by="timestamp", ascending=True)
    item_list = user_df["item_id"].tolist()
    
    if len(item_list) < 2:
        return None # User needs at least 2 interactions to have a train and test set.

    user_id = user_df["user_id"].iloc[0]
    all_interacted = set(item_list)

    # Train-test split
    train_positives = item_list[:-1]
    test_positive = item_list[-1]

    # Training Data Synthesis (positives = 1, negatives = 0)
    train_data = []
    
    # Paper uses specific set $I_i$ of interactions and $I \setminus I_i$ of negatives
    # For every positive interaction, we sample `train_negatives` random negatives
    for pos_item in train_positives:
        train_data.append((user_id, pos_item, 1.0))
        # sample negatives
        negs = generate_negative_samples(all_interacted, total_items, train_negatives)
        for neg_item in negs:
            train_data.append((user_id, neg_item, 0.0))

    # Test Data Synthesis
    test_negs = generate_negative_samples(all_interacted, total_items, test_negatives)

    return {
        "user_id": user_id,
        "train_data": train_data, # List of (user, item, label)
        "test_positive": test_positive, # Int
        "test_negatives": test_negs # List of Ints
    }

def build_federated_dataset(df: pd.DataFrame, total_items: int, train_negatives: int = 4) -> Dict[int, Dict]:
    """
    Given a global DataFrame of interactions, build a dictionary mapping user_id to their dataset split.
    This enables federated simulation where each user is a client.
    """
    logger.info("Building federated dataset. This might take a moment...")
    client_datasets = {}
    users = df["user_id"].unique()
    
    for user_id in users:
        user_df = df[df["user_id"] == user_id]
        processed = preprocess_user_data(user_df, total_items, train_negatives=train_negatives)
        if processed is not None:
            client_datasets[user_id] = processed
            
    logger.info(f"Successfully processed datasets for {len(client_datasets)} valid clients.")
    return client_datasets
