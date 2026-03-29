import os
import urllib.request
import zipfile
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data_raw")
ZIP_PATH = os.path.join(DATA_DIR, "ml-100k.zip")
CSV_PATH = os.path.join(DATA_DIR, "ml-100k", "u.data")

def download_movielens_100k():
    """Download and extract MovieLens 100K dataset if not exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if not os.path.exists(CSV_PATH):
        logger.info(f"Downloading ML-100K dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        logger.info("Download and extraction complete.")
    else:
        logger.info("ML-100K dataset already exists locally.")

def load_data() -> pd.DataFrame:
    """Load the raw u.data file into a pandas DataFrame."""
    download_movielens_100k()
    
    # u.data columns: user id | item id | rating | timestamp
    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(CSV_PATH, sep='\t', names=cols, engine='python')
    
    # 0-index the user and item IDs to use them in Embedding layers
    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1
    
    # Implicit feedback: keep ratings >= 1 as positive interactions (typically in ML-100k, all ratings 1-5 mean implicit interact)
    # So we just keep all rows since we treat presence as interaction.
    df['interaction'] = 1
    
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} interactions.")
    print(f"Num users: {df['user_id'].nunique()}, Num items: {df['item_id'].nunique()}")
