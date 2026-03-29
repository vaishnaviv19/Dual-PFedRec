import yaml
import uvicorn
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from src.server.aggregator import ServerAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PFedRec Server")

config = {}
aggregator = None

class AggregateRequest(BaseModel):
    client_updates: List[Dict[str, Any]] # list of state_dict lists

def tensor_to_list(state_dict):
    return {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}

def list_to_tensor(state_dict_list):
    return {k: torch.tensor(v) for k, v in state_dict_list.items()}

@app.on_event("startup")
async def startup_event():
    global config, aggregator
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    aggregator = ServerAggregator(
        num_items=config['num_items'],
        embedding_size=config['embedding_size']
    )
    logger.info("Server Initialized!")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/global_model")
def get_global_model():
    """Return the global embedding to the client simulator worker."""
    sd = aggregator.get_global_embedding_state()
    return {"status": "success", "global_embedding_state": tensor_to_list(sd)}

@app.post("/aggregate")
def aggregate_models(req: AggregateRequest):
    """
    Receive all fine-tuned local item embeddings, perform FedAvg,
    and return the updated global state_dict.
    """
    if not req.client_updates:
        return {"status": "empty"}
        
    # Convert lists back to tensors
    tensor_updates = [list_to_tensor(update) for update in req.client_updates]
    
    # Run FedAvg
    aggregator.aggregate(tensor_updates)
    
    # Send newly aggregated state back
    new_sd = aggregator.get_global_embedding_state()
    return {"status": "success", "global_embedding_state": tensor_to_list(new_sd)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
