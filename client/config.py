import os

class Config:
    # Client Settings
    CLIENT_ID = os.environ.get('CLIENT_ID', '1')
    SERVER_URL = os.environ.get('SERVER_URL', 'http://server:5000')
    
    # Model Settings (Paper Section 6.2)
    NUM_ITEMS = 1682  # MovieLens-100K
    EMBEDDING_SIZE = 32
    HIDDEN_SIZE = 64
    
    # Training Settings
    BATCH_SIZE = 256
    NEGATIVE_SAMPLING_RATIO = 4  # 4 negatives per positive
    LEARNING_RATE_SCORE = 0.01   # eta for score function
    LEARNING_RATE_ITEM = 0.001   # eta' for item embedding
    EPOCHS_LOCAL = 1             # Local epochs per round (E in Alg 1)
    
    # Federated Settings
    TOTAL_ROUNDS = 100
    ROUND_INTERVAL = 10  # Seconds between rounds
    
    # Privacy (Section 6.6)
    LAPLACIAN_NOISE_LAMBDA = 0.0  # Set > 0 for differential privacy