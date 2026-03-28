import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def prepare_movielens_data(raw_file, output_dir, n_clients=5):
    """
    Split MovieLens data into client-specific files
    Simulates non-IID federated setting
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(raw_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df = df[df['rating'] > 0]  # Keep positive interactions
    
    # Assign users to clients
    unique_users = df['user_id'].unique()
    user_groups = np.array_split(unique_users, n_clients)
    
    for i, users in enumerate(user_groups):
        client_df = df[df['user_id'].isin(users)]
        client_df.to_csv(f'{output_dir}/client_{i+1}.csv', index=False)
        print(f"✓ Created client_{i+1}.csv with {len(users)} users, {len(client_df)} interactions")
    
    print(f"\n✅ Data prepared for {n_clients} clients in {output_dir}")

if __name__ == '__main__':
    # Download MovieLens-100K from: https://grouplens.org/datasets/movielens/100k/
    # Extract to data/ml-100k/u.data
    prepare_movielens_data('ml-100k/u.data', '.', n_clients=5)