import os

class Config:
    # Server Settings
    SERVER_HOST = os.environ.get('SERVER_HOST', '0.0.0.0')
    SERVER_PORT = int(os.environ.get('SERVER_PORT', 5000))
    
    # Model Settings (Paper Section 6.2)
    NUM_ITEMS = int(os.environ.get('NUM_ITEMS', 1682))  # MovieLens-100K
    EMBEDDING_SIZE = int(os.environ.get('EMBEDDING_SIZE', 32))
    
    # Federated Learning Settings
    TOTAL_ROUNDS = 100
    MIN_CLIENTS_PER_ROUND = 2  # Wait for at least 2 clients before aggregating
    