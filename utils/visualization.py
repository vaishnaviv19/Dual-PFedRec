# utils/visualization.py
"""
Visualization Utilities for Research Analysis
- t-SNE embedding visualization (Paper Figure 2)
- Training progress plots
- Metric comparison charts
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import List, Dict, Optional
import seaborn as sns


def visualize_embeddings_tsne(embeddings: torch.Tensor, 
                             positive_items: List[int],
                             negative_items: List[int],
                             title: str = "Item Embedding Visualization",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    t-SNE visualization of item embeddings (Paper Figure 2)
    
    Shows separation between positive (interacted) and negative items
    after personalization vs before.
    
    Args:
        embeddings: (num_items, embedding_dim) tensor
        positive_items: List of item IDs user interacted with
        negative_items: List of sampled negative item IDs
        title: Plot title
        save_path: If provided, save figure to this path
        
    Returns:
        fig: Matplotlib figure object
    """
    # Convert to numpy and select items to visualize
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # Combine items for visualization
    all_items = positive_items + negative_items
    labels = [1] * len(positive_items) + [0] * len(negative_items)
    
    if len(all_items) == 0:
        raise ValueError("No items to visualize")
    
    # Extract embeddings for selected items
    item_embeddings = embeddings_np[np.array(all_items)]
    
    # Apply t-SNE for 2D visualization
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, 
               random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(item_embeddings)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot positive items (blue)
    pos_mask = np.array(labels) == 1
    ax.scatter(embeddings_2d[pos_mask, 0], embeddings_2d[pos_mask, 1],
              c='#2E86AB', label='Positive (Interacted)', 
              alpha=0.7, s=50, edgecolors='white')
    
    # Plot negative items (purple)
    neg_mask = np.array(labels) == 0
    ax.scatter(embeddings_2d[neg_mask, 0], embeddings_2d[neg_mask, 1],
              c='#A23B72', label='Negative (Sampled)',
              alpha=0.7, s=50, edgecolors='white')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    return fig


def plot_training_progress(rounds: List[int], 
                          train_losses: List[float],
                          hr_scores: List[float],
                          ndcg_scores: List[float],
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training progress over federated rounds
    
    Args:
        rounds: List of round numbers
        train_losses: Average training loss per round
        hr_scores: HR@10 scores per round
        ndcg_scores: NDCG@10 scores per round
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure with 3 subplots
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Training Loss
    axes[0].plot(rounds, train_losses, 'b-', linewidth=2, label='Training Loss')
    axes[0].set_xlabel('Federated Round', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training Loss Over Rounds', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Hit Ratio @ 10
    axes[1].plot(rounds, hr_scores, 'g-', linewidth=2, label='HR@10')
    axes[1].set_xlabel('Federated Round', fontsize=11)
    axes[1].set_ylabel('Hit Ratio @ 10', fontsize=11)
    axes[1].set_title('Hit Ratio Over Rounds', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim([0, 1])
    
    # Plot 3: NDCG @ 10
    axes[2].plot(rounds, ndcg_scores, 'r-', linewidth=2, label='NDCG@10')
    axes[2].set_xlabel('Federated Round', fontsize=11)
    axes[2].set_ylabel('NDCG @ 10', fontsize=11)
    axes[2].set_title('NDCG Over Rounds', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training progress to {save_path}")
    
    return fig


def compare_methods_results(results_df, save_path: Optional[str] = None):
    """
    Generate comparison table/plot for different methods (Paper Table 2)
    
    Args:
        results_df: DataFrame with columns: method, dataset, hr@10, ndcg@10
        save_path: Optional path to save
    """
    # Create grouped bar plot
    import pandas as pd
    
    datasets = results_df['dataset'].unique()
    methods = results_df['method'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # HR@10 comparison
    for i, dataset in enumerate(datasets[:2]):  # Show first 2 datasets
        subset = results_df[results_df['dataset'] == dataset]
        x = np.arange(len(methods))
        width = 0.35
        
        axes[0].bar(x - width/2 + i*0.1, 
                   subset['hr@10'].values, 
                   width, label=dataset, alpha=0.8)
    
    axes[0].set_xlabel('Method', fontsize=11)
    axes[0].set_ylabel('HR@10', fontsize=11)
    axes[0].set_title('Hit Ratio @ 10 Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # NDCG@10 comparison
    for i, dataset in enumerate(datasets[:2]):
        subset = results_df[results_df['dataset'] == dataset]
        x = np.arange(len(methods))
        width = 0.35
        
        axes[1].bar(x - width/2 + i*0.1,
                   subset['ndcg@10'].values,
                   width, label=dataset, alpha=0.8)
    
    axes[1].set_xlabel('Method', fontsize=11)
    axes[1].set_ylabel('NDCG@10', fontsize=11)
    axes[1].set_title('NDCG @ 10 Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig