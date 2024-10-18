import numpy as np


def evaluate_map(distmat, query_labels, gallery_labels):
    num_queries = query_labels.size(0)
    aps = []
    for i in range(num_queries):
        # Get query details
        q_label = query_labels[i].item()
        q_dist = distmat[i]
        # Sort distances and get indices
        indices = np.argsort(q_dist)
        # Compare gallery labels with query label
        matches = (gallery_labels[indices] == q_label).numpy()
        # Compute Average Precision
        ap = compute_average_precision(matches)
        aps.append(ap)
    mAP = np.mean(aps)
    return mAP

def compute_average_precision(matches):
    # Calculate AP for a single query
    num_relevant = matches.sum()
    if num_relevant == 0:
        return 0.0
    cumulative_hits = np.cumsum(matches)
    precision_at_k = cumulative_hits / (np.arange(len(matches)) + 1)
    ap = (precision_at_k * matches).sum() / num_relevant
    return ap