import yaml
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

from src.data.dataset import load_data
from src.data.preprocess import build_federated_dataset
from src.models.pfedrec import PFedRec
from src.client.trainer import ClientTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PFedRec Client Worker")

# Globals
config = {}
client_datasets = {}
client_models = {}

class TrainRequest(BaseModel):
    client_ids: List[int]
    global_embedding_state: Dict[str, Any] # nested list essentially, representing torch tensor

class EvalRequest(BaseModel):
    client_ids: List[int]

def tensor_to_list(state_dict):
    return {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}

def list_to_tensor(state_dict_list):
    return {k: torch.tensor(v) for k, v in state_dict_list.items()}

@app.on_event("startup")
async def startup_event():
    global config, client_datasets, client_models
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    df = load_data()
    client_datasets = build_federated_dataset(df, config['num_items'], train_negatives=config['num_negatives'])
    
    # Initialize a personalized model for each valid client
    for client_id in client_datasets.keys():
        client_models[client_id] = PFedRec(
            num_items=config['num_items'],
            embedding_size=config['embedding_size'],
            client_layers=config['client_layers']
        )
    logger.info(f"Initialized client models for {len(client_models)} users.")

@app.get("/health")
def health():
    return {"status": "healthy", "num_clients": len(client_models)}

@app.post("/train_round")
def train_round(req: TrainRequest):
    global_sd = list_to_tensor(req.global_embedding_state)
    updates = {}
    
    for client_id in req.client_ids:
        if client_id not in client_models:
            continue
            
        model = client_models[client_id]
        
        # Load global item embeddings
        model.load_global_embeddings(global_sd)
        
        # Setup trainer
        trainer = ClientTrainer(
            user_id=client_id,
            train_data=client_datasets[client_id]['train_data'],
            test_data=client_datasets[client_id],
            config=config,
            model=model
        )
        
        # Train (Steps 1 and 2)
        trainer.train_dual_personalization()
        
        # Collect fine-tuned embeddings to send back
        updates[str(client_id)] = tensor_to_list(model.get_local_embeddings())
        
    return {"status": "success", "client_updates": updates}

@app.post("/evaluate_round")
def evaluate_round(req: EvalRequest):
    metrics = {}
    total_hr = 0.0
    total_ndcg = 0.0
    count = 0
    
    for client_id in req.client_ids:
        if client_id not in client_models:
            continue
            
        model = client_models[client_id]
        trainer = ClientTrainer(
            user_id=client_id,
            train_data=client_datasets[client_id]['train_data'],
            test_data=client_datasets[client_id],
            config=config,
            model=model
        )
        
        hr, ndcg = trainer.evaluate()
        metrics[str(client_id)] = {"hr": hr, "ndcg": ndcg}
        
        total_hr += hr
        total_ndcg += ndcg
        count += 1
        
    if count == 0:
        return {"status": "empty"}
        
    return {
        "status": "success",
        "avg_hr_10": total_hr / count,
        "avg_ndcg_10": total_ndcg / count,
        "client_metrics": metrics
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
