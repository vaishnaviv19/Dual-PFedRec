"""server/main.py
Simple FastAPI server for federated aggregation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# make sure /app is on path when running `python server/main.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.aggregator import FederatedAggregator
from server.config import ServerConfig
from utils.logger import setup_logger, get_logger
from utils.privacy import add_laplacian_noise


app = FastAPI(
    title="PFedRec Server",
    description="Federated aggregation server",
    version="1.0.0",
)

config = ServerConfig.from_yaml("config.yaml")
setup_logger(level=config.log_level, log_dir=config.log_dir, name="pfedrec.server")
logger = get_logger("pfedrec.server")

aggregator = FederatedAggregator(
    num_items=config.num_items,
    embedding_dim=config.embedding_dim,
    aggregation_method=config.aggregation_method,
)


class ClientUpdatePayload(BaseModel):
    client_id: str
    round: int
    embedding: list
    num_samples: int = 1
    metrics: Optional[Dict[str, Any]] = None


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "round": aggregator.current_round,
        "pending_updates": len(aggregator.client_updates),
    }


@app.get("/api/v1/global_embedding")
async def get_global_embedding(client_id: str) -> Dict[str, Any]:
    try:
        embedding = aggregator.get_global_embedding()
        return {
            "success": True,
            "client_id": client_id,
            "round": aggregator.current_round,
            "embedding": embedding.tolist(),
            "shape": list(embedding.shape),
        }
    except Exception as e:
        logger.error(f"global_embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/client_update")
async def receive_client_update(update: ClientUpdatePayload) -> Dict[str, Any]:
    try:
        client_id = str(update.client_id)
        round_num = int(update.round)
        embedding_data = update.embedding
        num_samples = int(update.num_samples)

        embedding = torch.tensor(embedding_data, dtype=torch.float32)

        if config.privacy_enabled and config.ldp_lambda > 0:
            embedding = add_laplacian_noise(embedding, config.ldp_lambda)

        accepted = aggregator.receive_update(
            client_id=client_id,
            embedding=embedding,
            num_samples=num_samples,
            round_num=round_num,
        )
        if not accepted:
            return {"success": False, "error": "Update rejected: embedding shape mismatch"}

        new_embedding = aggregator.aggregate(min_clients=config.min_clients_per_round)
        return {
            "success": True,
            "client_id": client_id,
            "round": aggregator.current_round,
            "aggregated": new_embedding is not None,
            "pending_clients": len(aggregator.client_updates),
        }
    except Exception as e:
        logger.error(f"client_update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_stats() -> Dict[str, Any]:
    return aggregator.get_stats()


@app.get("/api/v1/metrics")
async def get_client_metrics(client_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Proxy endpoint so client metrics appear in SERVER Swagger UI.

    - If client_id is provided: return that client's metrics
    - If client_id is omitted: return metrics for all configured clients
    """
    num_clients = int(os.environ.get("NUM_CLIENTS", "3"))
    timeout = float(config.timeout)

    async def _fetch_one(cid: int) -> Dict[str, Any]:
        url = f"http://client_{cid}:8001/api/v1/metrics"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                return {"client_id": cid, "success": True, "data": data}
        except Exception as e:
            return {"client_id": cid, "success": False, "error": str(e)}

    if client_id is not None:
        if client_id < 1:
            raise HTTPException(status_code=400, detail="client_id must be >= 1")
        result = await _fetch_one(client_id)
        if not result["success"]:
            raise HTTPException(status_code=502, detail=result["error"])
        return {
            "success": True,
            "source": "client",
            "client_id": client_id,
            "metrics": result["data"].get("metrics", {}),
            "training_active": result["data"].get("training_active", False),
        }

    results = []
    hr_vals = []
    ndcg_vals = []
    for cid in range(1, num_clients + 1):
        one = await _fetch_one(cid)
        results.append(one)
        if one["success"]:
            metrics = one["data"].get("metrics", {})
            hr = metrics.get("hr@10")
            ndcg = metrics.get("ndcg@10")
            if isinstance(hr, (int, float)):
                hr_vals.append(float(hr))
            if isinstance(ndcg, (int, float)):
                ndcg_vals.append(float(ndcg))

    return {
        "success": True,
        "source": "clients",
        "num_clients": num_clients,
        "results": results,
        "average_metrics": {
            "hr@10": (sum(hr_vals) / len(hr_vals)) if hr_vals else None,
            "ndcg@10": (sum(ndcg_vals) / len(ndcg_vals)) if ndcg_vals else None,
        },
    }


@app.post("/api/v1/start_training")
async def start_training(client_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Proxy endpoint so training can be started from SERVER Swagger UI.

    - If client_id is provided: start that client's training
    - If client_id is omitted: start all configured clients
    """
    num_clients = int(os.environ.get("NUM_CLIENTS", "3"))
    timeout = float(config.timeout)

    async def _start_one(cid: int) -> Dict[str, Any]:
        url = f"http://client_{cid}:8001/api/v1/start_training"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url)
                resp.raise_for_status()
                return {"client_id": cid, "success": True, "data": resp.json()}
        except Exception as e:
            return {"client_id": cid, "success": False, "error": str(e)}

    if client_id is not None:
        if client_id < 1:
            raise HTTPException(status_code=400, detail="client_id must be >= 1")
        result = await _start_one(client_id)
        if not result["success"]:
            raise HTTPException(status_code=502, detail=result["error"])
        return {
            "success": True,
            "source": "client",
            "client_id": client_id,
            "response": result["data"],
        }

    results = []
    started = 0
    for cid in range(1, num_clients + 1):
        one = await _start_one(cid)
        results.append(one)
        if one["success"]:
            started += 1

    return {
        "success": True,
        "source": "clients",
        "num_clients": num_clients,
        "started_clients": started,
        "results": results,
    }


@app.post("/api/v1/reset")
async def reset_server() -> Dict[str, Any]:
    aggregator.reset()
    return {"success": True, "message": "Server state reset"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=config.debug,
    )