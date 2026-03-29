# utils/metrics.py
"""
Evaluation Metrics for Recommendation (Section 6.1)
- Hit Ratio @ K (HR@K)
- Normalized Discounted Cumulative Gain @ K (NDCG@K)
"""

import numpy as np
from typing import Dict, List, Union


def hit_ratio(ranked_items: np.ndarray, ground_truth: List[int], k: int = 10) -> float:
    """
    Hit Ratio @ K: Was the relevant item in top-K recommendations?
    
    HR@K = (1/|U|) Σᵤ I(rankᵤ ≤ K)
    
    Args:
        ranked_items: Items ranked by predicted score (descending)
        ground_truth: List of relevant item IDs for the user
        k: Cut-off rank
        
    Returns:
        hr: Hit Ratio score in [0, 1]
    """
    # Check if any ground truth item is in top-K
    top_k = ranked_items[:k]
    hits = sum(1 for item in ground_truth if item in top_k)
    return 1.0 if hits > 0 else 0.0


def ndcg(ranked_items: np.ndarray, ground_truth: List[int], k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain @ K
    
    NDCG@K = DCG@K / IDCG@K
    
    DCG@K = Σᵢ₌₁ᴷ (2^relᵢ - 1) / log₂(i + 1)
    
    For implicit feedback: relᵢ = 1 if item is relevant, else 0
    
    Args:
        ranked_items: Items ranked by predicted score (descending)
        ground_truth: List of relevant item IDs for the user
        k: Cut-off rank
        
    Returns:
        ndcg: NDCG score in [0, 1]
    """
    if not ground_truth:
        return 0.0
    
    # Create relevance array for ranked list
    relevance = np.array([1.0 if item in ground_truth else 0.0 
                         for item in ranked_items[:k]])
    
    if relevance.sum() == 0:
        return 0.0
    
    # DCG: Discounted Cumulative Gain
    discounts = np.log2(np.arange(len(relevance)) + 2)  # log2(i+2) for 1-indexed
    dcg = np.sum(relevance / discounts)
    
    # IDCG: Ideal DCG (all relevant items at top)
    ideal_relevance = np.zeros_like(relevance)
    ideal_relevance[:min(len(ground_truth), k)] = 1.0
    idcg = np.sum(ideal_relevance / discounts)
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_all_clients(client_metrics: List[Dict], 
                        metric_names: List[str] = None) -> Dict[str, float]:
    """
    Aggregate metrics across all clients
    
    Args:
        client_metrics: List of per-client metric dictionaries
        metric_names: Metrics to aggregate (default: all found)
        
    Returns:
        aggregated: Mean and std for each metric
    """
    if not client_metrics:
        return {}
    
    if metric_names is None:
        metric_names = list(client_metrics[0].keys())
    
    results = {}
    for metric in metric_names:
        values = [m[metric] for m in client_metrics if metric in m]
        if values:
            results[f"{metric}"] = np.mean(values)
            results[f"{metric}_std"] = np.std(values)
    
    return results