import os
import time
import torch
import requests
import logging
from model import PFedRecModel
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FederatedClient:
    
    def __init__(self, client_id, data_file, server_url):
        self.client_id = client_id
        self.data_file = data_file
        self.server_url = server_url
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load local data
        self.user_id, self.positive_items = self._load_data()
        
        # Initialize model
        self.model = PFedRecModel(
            num_items=Config.NUM_ITEMS,
            embed_size=Config.EMBEDDING_SIZE,
            hidden_size=Config.HIDDEN_SIZE
        ).to(self.device)
        
        logger.info(f"Client {client_id} initialized with {len(self.positive_items)} items")
    
    def _load_data(self):
        """Load user interaction data"""
        import pandas as pd
        df = pd.read_csv(self.data_file)
        users = df['user_id'].unique()
        if len(users) != 1:
            raise ValueError("Client data must belong to single user")
        return users[0], df['item_id'].values
    
    def get_global_model(self):
        """Download global item embedding from server"""
        try:
            resp = requests.get(
                f'{self.server_url}/get_model',
                params={'client_id': self.client_id},
                timeout=30
            )
            data = resp.json()
            if data['status'] == 'success':
                embedding = torch.tensor(data['embedding']).to(self.device)
                self.model.load_item_embedding(embedding)
                return data['round']
        except Exception as e:
            logger.error(f"Failed to get model: {e}")
        return None
    
    def train_local(self):
        """
        Dual Personalization Training (Algorithm 1, Lines 6-11)
        Step 1: Update Score Function (theta_s) - stays local
        Step 2: Finetune Item Embedding (theta_m) - sent to server
        """
        self.model.train()
        pos_items = torch.tensor(self.positive_items).to(self.device)
        
        for epoch in range(Config.EPOCHS_LOCAL):
            # --- Step 1: Update Score Function (Section 4.2) ---
            self.model.item_embedding.weight.requires_grad = False
            self.model.score_function.requires_grad = True
            
            for i in range(0, len(pos_items), Config.BATCH_SIZE):
                batch_pos = pos_items[i:i+Config.BATCH_SIZE]
                batch_neg = self._sample_negatives(batch_pos, len(batch_pos))
                
                loss = self._compute_loss(batch_pos, batch_neg)
                
                self.model.optimizer_score.zero_grad()
                loss.backward()
                self.model.optimizer_score.step()
            
            # --- Step 2: Finetune Item Embedding (Section 4.2) ---
            self.model.item_embedding.weight.requires_grad = True
            self.model.score_function.requires_grad = False
            
            for i in range(0, len(pos_items), Config.BATCH_SIZE):
                batch_pos = pos_items[i:i+Config.BATCH_SIZE]
                batch_neg = self._sample_negatives(batch_pos, len(batch_pos))
                
                loss = self._compute_loss(batch_pos, batch_neg)
                
                self.model.optimizer_item.zero_grad()
                loss.backward()
                self.model.optimizer_item.step()
    
    def _sample_negatives(self, positive_batch, num_pos):
        """Sample negative items (Eq. 8)"""
        all_items = set(range(Config.NUM_ITEMS))
        positive_set = set(positive_batch.cpu().numpy())
        negative_pool = list(all_items - positive_set)
        
        num_neg = num_pos * Config.NEGATIVE_SAMPLING_RATIO
        negatives = torch.tensor(
            np.random.choice(negative_pool, size=num_neg, replace=False)
        ).to(self.device)
        return negatives
    
    def _compute_loss(self, pos_items, neg_items):
        """Binary Cross-Entropy Loss (Eq. 7)"""
        pos_pred = self.model(pos_items)
        neg_pred = self.model(neg_items)
        
        loss = torch.nn.BCELoss()(
            pos_pred, torch.ones_like(pos_pred)
        ) + torch.nn.BCELoss()(
            neg_pred, torch.zeros_like(neg_pred)
        )
        return loss
    
    def send_update(self, round_num):
        """Upload finetuned item embedding to server"""
        embedding = self.model.get_item_embedding().cpu()
        
        # Add Differential Privacy Noise (Section 6.6)
        if Config.LAPLACIAN_NOISE_LAMBDA > 0:
            noise = torch.from_numpy(
                np.random.laplace(0, Config.LAPLACIAN_NOISE_LAMBDA, embedding.shape)
            ).float()
            embedding = embedding + noise
        
        try:
            resp = requests.post(
                f'{self.server_url}/update_model',
                json={
                    'client_id': self.client_id,
                    'embedding': embedding.tolist(),
                    'num_samples': len(self.positive_items),
                    'round': round_num
                },
                timeout=30
            )
            data = resp.json()
            logger.info(f"✓ Update sent (Round {data['round']})")
            return True
        except Exception as e:
            logger.error(f"Failed to send update: {e}")
            return False
    
    def run(self):
        """Main client loop"""
        logger.info(f"🚀 Client {self.client_id} starting...")
        
        for round_num in range(1, Config.TOTAL_ROUNDS + 1):
            # Wait for server to be ready
            time.sleep(5)
            
            # Get global model
            current_round = self.get_global_model()
            if current_round is None:
                continue
            
            # Train locally
            self.train_local()
            
            # Send update
            self.send_update(current_round)
            
            # Wait before next round
            time.sleep(Config.ROUND_INTERVAL)
        
        logger.info(f"✅ Client {self.client_id} completed all rounds")

if __name__ == '__main__':
    client_id = os.environ.get('CLIENT_ID', '1')
    data_file = os.environ.get('DATA_FILE', '/app/data/client_1.csv')
    server_url = os.environ.get('SERVER_URL', 'http://server:5000')
    
    client = FederatedClient(client_id, data_file, server_url)
    client.run()