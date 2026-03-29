# client/main.py
"""
PFedRec Client - FastAPI Entry Point
Implements local training with dual personalization (Algorithm 1)
"""

import os
import sys
import asyncio
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client.model import PFedRecModel
from client.config import ClientConfig
from data.loader import load_client_data  # ✅ Fixed: client-specific loader
from data.sampler import NegativeSampler
from utils.metrics import hit_ratio, ndcg
from utils.logger import setup_logger, get_logger
from utils.privacy import add_laplacian_noise  # ✅ Added: privacy

# Initialize FastAPI app
app = FastAPI(
    title="PFedRec Client",
    description="Client for Dual Personalization Federated Recommendation",
    version="1.0.0"
)

# Load configuration
config = ClientConfig.from_yaml("config.yaml")
logger = get_logger(__name__)

# Global state
client_id: Optional[str] = None
model: Optional[PFedRecModel] = None
train_data: Optional[Dict] = None  # ✅ Fixed: syntax
test_data: Optional[Dict] = None    # ✅ Fixed: syntax
negative_sampler: Optional[NegativeSampler] = None
server_url: Optional[str] = None
training_active: bool = False


class ClientUpdateRequest(BaseModel):
    """Request schema for receiving server embedding"""
    client_id: str
    round: int
    embedding: List[List[float]]


class ClientUpdateResponse(BaseModel):
    """Response schema for sending update to server"""
    success: bool
    round: int
    samples: int
    loss: float


@app.on_event("startup")
async def startup_event():
    """Initialize client on startup"""
    global client_id, model, train_data, test_data, negative_sampler, server_url
    
    # Get client ID from environment
    client_id = os.environ.get("CLIENT_ID", "1")
    server_url = os.environ.get("SERVER_URL", "http://server:8000")
    data_file = os.environ.get("DATA_FILE", f"data/client_{client_id}.csv")
    
    logger.info(f"🚀 Client {client_id} starting, server: {server_url}")
    
    # ✅ Load data using correct function
    logger.info(f"Loading data from {data_file}")
    user_id, interactions = load_client_data(data_file)  # Returns (user_id, item_array)
    
    # Split data (leave-one-out) - simple: last item = test
    if len(interactions) > 1:
        train_items = interactions[:-1]
        test_items = [interactions[-1]]
    else:
        train_items = interactions
        test_items = []
    
    train_data = {
        "user_id": user_id,
        "items": np.array(train_items),
        "all_items": set(range(config.num_items))
    }
    test_data = {
        "items": np.array(test_items) if test_items else np.array([])
    }
    
    # Initialize negative sampler (Eq. 8)
    negative_sampler = NegativeSampler(
        num_items=config.num_items,
        ratio=config.negative_sampling_ratio
    )
    
    # Initialize model
    model = PFedRecModel(
        num_items=config.num_items,
        embedding_dim=config.embedding_dim,
        score_hidden_dims=config.score_hidden_dims
    )
    
    # Download initial global embedding
    await download_global_embedding()
    
    logger.info(f"✓ Client {client_id} initialized with "
               f"{len(train_data['items'])} training items")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "client_id": client_id,
        "model_initialized": model is not None,
        "data_loaded": train_data is not None
    }


@app.post("/api/v1/start_training")
async def start_training():
    """Start local training loop"""
    global training_active
    
    if training_active:
        return {"success": False, "error": "Training already active"}
    
    if model is None or train_data is None:
        raise HTTPException(status_code=503, detail="Client not initialized")
    
    training_active = True
    # ✅ Fixed: Use asyncio.create_task for async background task
    asyncio.create_task(run_local_training_loop())
    
    return {"success": True, "message": f"Training started for client {client_id}"}


async def run_local_training_loop():
    """Execute local training with dual personalization (Algorithm 1)"""
    global training_active
    
    logger.info(f"🎯 Client {client_id} starting local training")
    
    # Import trainer here to avoid circular imports
    from client.trainer import PFedRecTrainer
    trainer = PFedRecTrainer(
        model=model,
        config=config.to_dict(),
        negative_sampler=negative_sampler
    )
    
    for round_num in range(1, config.total_rounds + 1):
        try:
            # Download global embedding from server
            await download_global_embedding()
            
            # Train locally (Algorithm 1: ClientUpdate)
            updated_embedding, metrics = trainer.train_local(
                positive_items=train_data["items"],
                all_items=train_data["all_items"]
            )
            
            # ✅ Evaluate on test set (if available)
            eval_metrics = {}
            if len(test_data["items"]) > 0 and round_num % config.eval_every == 0:
                eval_metrics = _evaluate_local(
                    test_items=test_data["items"],
                    train_items=set(train_data["items"]),
                    all_items=train_data["all_items"],
                    k=10
                )
                logger.info(f"Round {round_num} eval: {eval_metrics}")
            
            # ✅ Send update to server (with optional privacy noise)
            await send_update_to_server(
                embedding=updated_embedding,
                num_samples=metrics["samples"],
                round_num=round_num,
                metrics={**metrics, **eval_metrics}
            )
            
            # Log progress
            if round_num % 10 == 0:
                logger.info(f"✓ Client {client_id} completed round {round_num}: "
                           f"loss={metrics['avg_loss']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in round {round_num}: {e}", exc_info=True)
            continue
    
    training_active = False
    logger.info(f"✅ Client {client_id} training complete")


async def download_global_embedding():
    """Download global item embedding from server"""
    global model
    
    if model is None:
        return
    
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.get(
                f"{server_url}/api/v1/global_embedding",
                params={"client_id": client_id}
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("success"):
                embedding = torch.tensor(data["embedding"])
                model.load_item_embedding_weights(embedding)
                logger.debug(f"Downloaded global embedding (round {data['round']})")
                
    except httpx.RequestError as e:
        logger.warning(f"Failed to download embedding: {e}")
    except Exception as e:
        logger.error(f"Unexpected error downloading embedding: {e}")


async def send_update_to_server(embedding: torch.Tensor, 
                               num_samples: int,
                               round_num: int,
                               metrics: Dict):
    """Send fine-tuned embedding to server with optional privacy noise"""
    try:
        # ✅ Add Local Differential Privacy noise if enabled (Section 6.6)
        if config.enable_ldp and config.ldp_lambda > 0:
            embedding = add_laplacian_noise(
                embedding, 
                lambda_param=config.ldp_lambda,
                device=embedding.device
            )
            logger.debug(f"Added LDP noise (λ={config.ldp_lambda}) before upload")
        
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                f"{server_url}/api/v1/client_update",
                json={
                    "client_id": client_id,
                    "round": round_num,
                    "embedding": embedding.tolist(),
                    "num_samples": num_samples,
                    "metrics": metrics
                }
            )
            response.raise_for_status()
            logger.debug(f"Sent update to server (round {round_num})")
            
    except httpx.RequestError as e:
        logger.warning(f"Failed to send update: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending update: {e}")


def _evaluate_local(test_items: np.ndarray,
                   train_items: set,
                   all_items: set,
                   k: int = 10) -> Dict[str, float]:
    """
    Local evaluation function (leave-one-out)
    
    Returns:
        metrics: {"hr@10": float, "ndcg@10": float}
    """
    if model is None or len(test_items) == 0:
        return {"hr@10": 0.0, "ndcg@10": 0.0}
    
    model.eval()
    test_item = test_items[0]  # Leave-one-out: single test item
    
    # Create candidate set: test item + 99 randomly sampled negatives
    candidates = [test_item]
    negative_pool = list(all_items - train_items - {test_item})
    
    if len(negative_pool) >= 99:
        candidates.extend(np.random.choice(negative_pool, size=99, replace=False))
    else:
        candidates.extend(negative_pool)
    
    # Get predictions
    candidate_tensor = torch.tensor(candidates, dtype=torch.long)
    with torch.no_grad():
        scores = model(candidate_tensor).cpu().numpy()
    
    # Rank by score (descending)
    ranked = np.array(candidates)[np.argsort(-scores)]
    
    # Compute metrics
    hr = hit_ratio(ranked, [test_item], k)
    ndcg_score = ndcg(ranked, [test_item], k)
    
    return {"hr@10": float(hr), "ndcg@10": float(ndcg_score)}


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get current client metrics"""
    if model is None or train_data is None:
        raise HTTPException(status_code=503, detail="Client not initialized")
    
    # Run evaluation
    metrics = _evaluate_local(
        test_items=test_data["items"],
        train_items=set(train_data["items"]),
        all_items=train_data["all_items"],
        k=10
    )
    
    return {
        "client_id": client_id,
        "metrics": metrics,
        "training_active": training_active
    }


@app.post("/api/v1/reset")
async def reset_client():
    """Reset client state for new experiment"""
    global model, training_active
    
    # Re-initialize model with random weights
    model = PFedRecModel(
        num_items=config.num_items,
        embedding_dim=config.embedding_dim,
        score_hidden_dims=config.score_hidden_dims
    )
    training_active = False
    
    # Download fresh global embedding
    await download_global_embedding()
    
    logger.info(f"🔄 Client {client_id} reset complete")
    return {"success": True, "message": "Client reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "client.main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=config.debug
    )