#!/usr/bin/env python3
# data/prepare.py
"""
Data Preparation Script
Prepares MovieLens-100K for federated learning experiments

Usage:
    python data/prepare.py --data-dir data/ml-100k --output-dir data --clients 100
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_movielens_100k, filter_users_by_interactions
from data.splitter import split_dataset_by_users


def prepare_client_files(df: pd.DataFrame, 
                        output_dir: str,
                        n_clients: int = None,
                        min_interactions: int = 20):
    """
    Create client-specific CSV files from full dataset
    
    Args:
        df: Full interaction DataFrame
        output_dir: Directory to save client files
        n_clients: Number of clients (None = all eligible users)
        min_interactions: Minimum interactions per user
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter users
    df_filtered = filter_users_by_interactions(df, min_interactions)
    
    # Split by users
    client_data = split_dataset_by_users(
        df_filtered, 
        n_clients=n_clients,
        min_interactions=0  # Already filtered
    )
    
    # Save each client's data
    saved = 0
    for client_id, client_df in client_data.items():
        output_file = output_path / f"client_{client_id}.csv"
        client_df.to_csv(output_file, index=False)
        saved += 1
        
        if saved <= 3:  # Log first few for verification
            print(f"✓ Created {output_file} with {len(client_df)} interactions")
    
    print(f"\n✅ Prepared {saved} client files in {output_dir}")
    return saved


def verify_data_integrity(data_dir: str):
    """Verify prepared data meets requirements"""
    data_path = Path(data_dir)
    
    client_files = list(data_path.glob("client_*.csv"))
    if not client_files:
        print(f"⚠️  No client files found in {data_dir}")
        return False
    
    print(f"📊 Found {len(client_files)} client files")
    
    # Check a sample file
    sample = pd.read_csv(client_files[0])
    print(f"📋 Sample file columns: {list(sample.columns)}")
    print(f"📋 Sample file shape: {sample.shape}")
    
    # Verify required columns
    required = ['user_id', 'item_id']
    if not all(col in sample.columns for col in required):
        print(f"❌ Missing required columns: {required}")
        return False
    
    print("✅ Data integrity check passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare MovieLens data for PFedRec")
    parser.add_argument("--data-dir", type=str, default="data/ml-100k",
                       help="Path to MovieLens-100K directory")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory for client files")
    parser.add_argument("--clients", type=int, default=None,
                       help="Number of clients to create (default: all users)")
    parser.add_argument("--min-interactions", type=int, default=20,
                       help="Minimum interactions per user")
    
    args = parser.parse_args()
    
    print(f"🚀 Preparing data from {args.data_dir}")
    
    # Load raw data
    try:
        df = load_movielens_100k(args.data_dir)
        print(f"✓ Loaded {len(df)} interactions from {len(df['user_id'].unique())} users")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n📥 Download MovieLens-100K:")
        print("   wget https://files.grouplens.org/datasets/movielens/ml-100k.zip")
        print("   unzip ml-100k.zip -d data/")
        return 1
    
    # Prepare client files
    n_saved = prepare_client_files(
        df, 
        args.output_dir, 
        args.clients,
        args.min_interactions
    )
    
    if n_saved == 0:
        print("❌ No client files created")
        return 1
    
    # Verify
    verify_data_integrity(args.output_dir)
    
    print(f"\n🎯 Data preparation complete!")
    print(f"   Ready to run: docker-compose up --build")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())