# Dual Personalization on Federated Recommendation (PFedRec)

A complete, research-grade PyTorch & FastAPI implementation.

## Features ✨

- **Architecture:** Server-Client Federated Learning setup using REST APIs for aggregation and local simulations.
- **Algorithm:** Strict adherence to Algorithm 1 from the paper.
  - Step 1: Update private `score_function` (MLP) with frozen user/item embeddings.
  - Step 2: Update local `item_embedding` with frozen `score_function`.
  - Step 3: FedAvg Server-side aggregation of fine-tuned item embeddings.
- **Dataset handling:** MovieLens-100K integrated, with automatic leave-one-out splits and 4-negative sampling per positive interaction.
- **Evaluation:** Real-time logging of HR@10 and NDCG@10.
- **Visualization:** `setup` includes t-SNE plot generators for verifying dual-personalization.
- **Baseline:** Includes `FedMF` baseline code.

## Requirements

- Docker & Docker Compose
- Python 3.12 (if running locally without Docker)
- Git Bash or equivalent terminal on Windows

## Local Setup Without Docker (Git Bash)

1. Open Git Bash and navigate into this directory.
2. Create virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Server (Terminal 1):
   ```bash
   uvicorn src.server.main:app --host 0.0.0.0 --port 8000
   ```
5. Start the Client Simulator Worker (Terminal 2):
   ```bash
   uvicorn src.client.main:app --host 0.0.0.0 --port 8001
   ```
   *(Wait up to 10-20 seconds on the first run as it downloads ML-100K and does the train test split generation).*
6. Run the Experiment Orchestrator (Terminal 3):
   ```bash
   python -m src.scripts.run_experiment
   ```
   *(This script will prompt the client and server to begin training rounds and print NDCG@10 / HR@10).*

## Docker Setup

To avoid GCC/build issues and dependency conflicts on Windows, use Docker Compose:

1. Open your terminal in this directory.
2. Build and run the containers:
   ```bash
   docker-compose up --build -d
   ```
3. Monitor logs to ensure dataset downloading finishes:
   ```bash
   docker-compose logs -f pfedrec-client-worker
   ```
4. Once the worker starts, run the orchestrator locally (or from within a container) to start training:
   ```bash
   docker exec -it pfedrec-client-worker python -m src.scripts.run_experiment
   ```
