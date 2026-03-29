import numpy as np

def hit_ratio_at_k(hits, k=10) -> float:
    """
    Computes Hit Ratio at K.
    `hits` is an array of 1s (hits) and 0s (misses).
    Since we only have 1 positive test item, HR@K is 1 if it is in the top K, else 0.
    """
    for item in hits[:k]:
        if item == 1.0:
            return 1.0
    return 0.0

def ndcg_at_k(hits, k=10) -> float:
    """
    Computes NDCG at K.
    `hits` is an array of 1s (hits) and 0s (misses).
    Since we only have 1 positive test item, Ideal DCG (IDCG) = 1.0.
    """
    for num, item in enumerate(hits[:k]):
        if item == 1.0:
            # Score is 1 / log2(rank + 1), where rank is 1-indexed.
            return 1.0 / np.log2(num + 2.0)
    return 0.0

def evaluate_predictions(scores, labels, k=10):
    """
    Given an array of predictions (scores) and ground truth (labels),
    compute HR@K and NDCG@K.
    """
    # Create combined array and sort by scores descending
    combined = list(zip(scores, labels))
    combined.sort(key=lambda x: x[0], reverse=True)
    
    # Extract the sorted labels to check where the `1` landed
    sorted_labels = [label for _, label in combined]
    
    hr = hit_ratio_at_k(sorted_labels, k)
    ndcg = ndcg_at_k(sorted_labels, k)
    
    return hr, ndcg
