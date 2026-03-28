from flask import Flask, request, jsonify
import torch
import numpy as np
import json
import os
from aggregator import FederatedAggregator
from config import Config

app = Flask(__name__)
aggregator = FederatedAggregator(Config)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "PFedRec Server"})

@app.route('/get_model', methods=['GET'])
def get_model():
    """Client downloads global item embedding"""
    client_id = request.args.get('client_id', 'unknown')
    embedding = aggregator.get_global_embedding()
    
    app.logger.info(f"Client {client_id} downloaded model")
    return jsonify({
        "embedding": embedding.tolist(),
        "round": aggregator.current_round,
        "status": "success"
    })

@app.route('/update_model', methods=['POST'])
def update_model():
    """Client uploads updated item embedding"""
    data = request.json
    client_id = data.get('client_id', 'unknown')
    embedding = torch.tensor(data['embedding'])
    num_samples = data.get('num_samples', 1)
    
    # Aggregate update
    aggregator.aggregate(client_id, embedding, num_samples)
    
    app.logger.info(f"Client {client_id} uploaded update (Round {aggregator.current_round})")
    return jsonify({
        "status": "success",
        "round": aggregator.current_round,
        "total_clients": aggregator.client_count
    })

@app.route('/status', methods=['GET'])
def status():
    """Get federated learning status"""
    return jsonify({
        "current_round": aggregator.current_round,
        "total_rounds": Config.TOTAL_ROUNDS,
        "registered_clients": aggregator.client_count,
        "embedding_size": Config.EMBEDDING_SIZE,
        "num_items": Config.NUM_ITEMS
    })

@app.route('/reset', methods=['POST'])
def reset():
    """Reset federated learning process"""
    aggregator.reset()
    return jsonify({"status": "reset_complete"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)