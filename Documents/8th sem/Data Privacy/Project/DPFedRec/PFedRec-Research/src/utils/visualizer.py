import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import logging

logger = logging.getLogger(__name__)

def plot_tsne_embeddings(global_embeds: np.ndarray, local_embeds: np.ndarray, client_id: int, save_dir="plots"):
    """
    Plots a t-SNE visualization comparing global item embeddings vs fine-tuned local item embeddings.
    Only visualizes the first 200 items to avoid clutter.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    logger.info(f"Generating t-SNE plot for Client {client_id}...")
    
    num_items = min(200, global_embeds.shape[0])
    
    # Take subset
    g_sub = global_embeds[:num_items]
    l_sub = local_embeds[:num_items]
    
    # Combine to fit t-SNE on same latent space
    combined = np.vstack([g_sub, l_sub])
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    reduced = tsne.fit_transform(combined)
    
    # Split back
    g_reduced = reduced[:num_items]
    l_reduced = reduced[num_items:]
    
    plt.figure(figsize=(10, 8))
    
    # Plot global embeddings
    plt.scatter(g_reduced[:, 0], g_reduced[:, 1], c='blue', alpha=0.5, label='Global Embeddings', marker='o')
    
    # Plot local fine-tuned embeddings
    plt.scatter(l_reduced[:, 0], l_reduced[:, 1], c='red', alpha=0.5, label='Local PFedRec Embeddings', marker='x')
    
    # Draw lines between global and local versions of the same item
    for i in range(num_items):
        plt.plot(
            [g_reduced[i, 0], l_reduced[i, 0]], 
            [g_reduced[i, 1], l_reduced[i, 1]], 
            'gray', alpha=0.2
        )
        
    plt.title(f"t-SNE Comparison for Client {client_id}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, f"tsne_client_{client_id}.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Plot saved to {save_path}")

if __name__ == "__main__":
    # Test script with random data
    dummy_global = np.random.randn(1682, 32)
    dummy_local = dummy_global + np.random.randn(1682, 32) * 0.5
    plot_tsne_embeddings(dummy_global, dummy_local, client_id=999)
