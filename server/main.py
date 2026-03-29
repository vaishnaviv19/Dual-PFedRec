# server/main.py
"""
PFedRec Federated Server - FastAPI Entry Point
Implements REST API for federated coordination
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import yaml
from pathlib import Path
from typing import List, Dict, Optional

from server.aggregator import FederatedAggregator
from server.config import ServerConfig
from utils.logger import setup_logger, get_logger
from utils.privacy import add_laplacian_noise

# Initialize FastAPI app
app = FastAPI(
    title="PFedRec Server",
    description="Federated Recommendation Server with Dual Personalization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config = ServerConfig.from_yaml("config.yaml")
logger = get_logger(__name__)

# Initialize aggregator
aggregator = FederatedAggregator(
    num_items=config.num_items,
    embedding_dim=config.embedding_dim,
    aggregation_method=config.aggregation_method
)

# Global state
training_active = False
round_metrics: List[Dict] = []


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup"""
    logger.info(f"🚀 PFedRec Server starting on {config.host}:{config.port}")
    logger.info(f"   Items: {config.num_items}, Embedding dim: {config.embedding_dim}")
    setup_logger(config.log_level, config.log_dir)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "pfedrec-server",
        "version": "1.0.0",
        "round": aggregator.current_round
    }


@app.get("/api/v1/global_embedding")
async def get_global_embedding(client_id: str):
    """
    Client endpoint: Download current global item embedding
    
    GET /api/v1/global_embedding?client_id=123
    """
    try:
        embedding = aggregator.get_global_embedding()
        
        logger.info(f"Client {client_id} downloaded global embedding "
                   f"(round {aggregator.current_round})")
        
        return {
            "success": True,
            "round": aggregator.current_round,
            "embedding": embedding.tolist(),  # Serialize for JSON
            "shape": list(embedding.shape)
        }
    except Exception as e:
        logger.error(f"Error serving embedding to client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/client_update")
async def receive_client_update(update: Dict):
    """
    Client endpoint: Upload fine-tuned item embedding
    
    POST /api/v1/client_update
    Body: {
        "client_id": "123",
        "round": 5,
        "embedding": [[...], ...],  # num_items × embedding_dim
        "num_samples": 150,
        "metrics": {"loss": 0.45}
    }
    """
    try:
        client_id = update.get("client_id")
        round_num = update.get("round")
        embedding_data = update.get("embedding")
        num_samples = update.get("num_samples", 1)
        
        if not all([client_id, round_num, embedding_data]):
            raise ValueError("Missing required fields")
        
        # Convert to tensor
        embedding = torch.tensor(embedding_data)
        
        # Apply differential privacy if enabled (Section 6.6)
        if config.privacy_enabled and config.ldp_lambda > 0:
            embedding = add_laplacian_noise(
                embedding, 
                lambda_param=config.ldp_lambda
            )
            logger.debug(f"Added LDP noise (λ={config.ldp_lambda}) for client {client_id}")
        
        # Receive update
        success = aggregator.receive_update(
            client_id=client_id,
            embedding=embedding,
            num_samples=num_samples,
            round_num=round_num
        )
        
        if not success:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Update rejected"}
            )
        
        # Attempt aggregation if enough clients
        new_embedding = aggregator.aggregate(min_clients=config.min_clients_per_round)
        
        response = {
            "success": True,
            "client_id": client_id,
            "round": aggregator.current_round,
            "aggregated": new_embedding is not None,
            "pending_clients": len(aggregator.client_updates)
        }
        
        logger.info(f"✓ Update from client {client_id} processed")
        return response
        
    except Exception as e:
        logger.error(f"Error processing client update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/status")
async def get_training_status():
    """Get current federated training status"""
    return {
        "active": training_active,
        "current_round": aggregator.current_round,
        "total_rounds": config.total_rounds,
        "aggregator_stats": aggregator.get_stats(),
        "config": {
            "clients_per_round": config.clients_per_round,
            "embedding_dim": config.embedding_dim
        }
    }


@app.post("/api/v1/start_training")
async def start_training(background_tasks: BackgroundTasks):
    """Start federated training loop"""
    global training_active
    
    if training_active:
        return {"success": False, "error": "Training already active"}
    
    training_active = True
    aggregator.reset()
    
    # Run training in background
    background_tasks.add_task(run_federated_training)
    
    return {"success": True, "message": "Training started"}


async def run_federated_training():
    """Background task: Execute federated training loop"""
    global training_active, round_metrics
    
    logger.info(f"🎯 Starting federated training: {config.total_rounds} rounds")
    
    for round_num in range(1, config.total_rounds + 1):
        aggregator.current_round = round_num
        
        # In real deployment: sample clients, send embeddings, wait for updates
        # For simulation: aggregation happens when clients send updates
        
        # Log progress
        if round_num % config.eval_every == 0:
            stats = aggregator.get_stats()
            logger.info(f"📊 Round {round_num}: {stats}")
            
            # Save checkpoint
            if config.save_checkpoints and round_num % config.checkpoint_every == 0:
                save_checkpoint(round_num)
    
    training_active = False
    logger.info("✅ Federated training complete")


def save_checkpoint(round_num: int):
    """Save model checkpoint"""
    checkpoint_path = Path(config.checkpoint_dir) / f"round_{round_num}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "round": round_num,
        "embedding": aggregator.get_global_embedding(),
        "config": config.dict()
    }, checkpoint_path)
    
    logger.info(f"💾 Checkpoint saved: {checkpoint_path}")


@app.get("/api/v1/metrics")
async def get_metrics(round_start: int = 0, round_end: Optional[int] = None):
    """Get evaluation metrics for specified rounds"""
    # In production: query metrics database
    # For now: return placeholder
    return {
        "rounds": list(range(round_start, round_end or aggregator.current_round)),
        "hr@10": [],  # Populate from evaluation
        "ndcg@10": []
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=config.debug
    )