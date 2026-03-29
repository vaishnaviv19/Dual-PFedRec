import requests
import yaml
import time
import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExperimentOrchestrator")

SERVER_URL = "http://localhost:8000"
CLIENT_URL = "http://localhost:8001"

def wait_for_services():
    """Wait until both server and client worker APIs are up."""
    logger.info("Waiting for FastAPI services to start...")
    while True:
        try:
            res_s = requests.get(f"{SERVER_URL}/health", timeout=2)
            res_c = requests.get(f"{CLIENT_URL}/health", timeout=2)
            if res_s.status_code == 200 and res_c.status_code == 200:
                logger.info(f"Services are UP! Total simulated clients: {res_c.json().get('num_clients')}")
                # We need the list of clients. So we just assume 0 to count-1 or we could add an endpoint
                # to get valid client IDs. Assuming 0 to count-1 valid users based on data filtering:
                return res_c.json().get('num_clients')
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    num_valid_clients = wait_for_services()
    if num_valid_clients is None or num_valid_clients == 0:
        logger.error("No valid clients found. Exiting.")
        return

    # In ML-100K, valid clients are dynamically found by preprocess, but typically max 943.
    # We will just assume valid_clients are [0, 1, 2, ..., num_valid_clients-1] for simulator simplicity.
    # BUT, actually `preprocess.py` creates a dict with valid `user_id` mapped.
    # A cleaner way is to make an endpoint to fetch valid active clients.
    # For now, we will query them directly.
    # Wait, the simplest way is to fetch the config['num_users'] and let the worker ignore invalid ones.
    all_client_ids = list(range(config['num_users']))
    
    global_rounds = config['global_rounds']
    fraction_fit = config['fraction_fit']
    clients_per_round = max(config['min_clients'], int(fraction_fit * len(all_client_ids)))
    
    logger.info(f"Starting Federated Training: {global_rounds} Rounds, {clients_per_round} Clients/Round")
    
    for r in range(1, global_rounds + 1):
        logger.info(f"--- Round {r} ---")
        
        # 1. Server: Get current global embeddings
        res = requests.get(f"{SERVER_URL}/global_model")
        global_sd = res.json()["global_embedding_state"]
        
        # 2. Sample clients
        selected_clients = np.random.choice(all_client_ids, clients_per_round, replace=False).tolist()
        logger.info(f"Selected {len(selected_clients)} clients.")
        
        # 3. Client Worker: Train local dual personalization
        logger.info(f"Triggering local step 1 & 2 for {len(selected_clients)} clients...")
        train_res = requests.post(f"{CLIENT_URL}/train_round", json={
            "client_ids": selected_clients,
            "global_embedding_state": global_sd
        })
        client_updates = train_res.json()["client_updates"]
        logger.info(f"Received fine-tuned item embeddings from {len(client_updates)} clients.")
        
        # 4. Server: Aggregate
        # Convert the dict format `{"0": {...}, "1": {...}}` into a list of updates `[{...}, {...}]`
        updates_list = list(client_updates.values())
        logger.info("Aggregating embeddings on server...")
        agg_res = requests.post(f"{SERVER_URL}/aggregate", json={
            "client_updates": updates_list
        })
        
        # 5. Evaluate all clients periodically
        if r % 5 == 0 or r == global_rounds:
            logger.info("Evaluating all clients...")
            eval_res = requests.post(f"{CLIENT_URL}/evaluate_round", json={
                "client_ids": all_client_ids
            })
            if eval_res.json().get('status') == 'success':
                metrics = eval_res.json()
                logger.info(f"Round {r} Evaluation | HR@10: {metrics['avg_hr_10']:.4f} | NDCG@10: {metrics['avg_ndcg_10']:.4f}")

if __name__ == "__main__":
    main()
