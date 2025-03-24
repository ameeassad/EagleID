# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import torch
import pandas as pd

from pytorch_metric_learning import distances, losses, miners




def get_all_embeddings(extractor, dataloader):
    all_embeddings = []
    # all_labels = []
    for batch in dataloader:
        inputs, labels = batch
        embeddings = extractor.get_embeddings(inputs)
        all_embeddings.append(embeddings)
        # all_labels.append(labels)
    all_embeddings = torch.cat(all_embeddings)
    # all_labels = torch.cat(all_labels)
    return all_embeddings #, all_labels

def get_all_embeddings_val(extractor, dataloader, query):
    all_embeddings = []
    all_labels = []
    
    for batch in dataloader:
        inputs, labels, is_query = batch
        embeddings = extractor.get_embeddings(inputs)
        
        # Filter based on is_query if `query` is True
        if query:
            mask = is_query.bool()
            embeddings = embeddings[mask]
            labels = labels[mask]
        
        # Handle scalar labels by converting to tensors
        labels = [torch.tensor(label) if isinstance(label, int) else label for label in labels]
        labels = torch.stack(labels)  # Stack to ensure consistency

        all_embeddings.append(embeddings)
        all_labels.append(labels)
    
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels) if all_labels else torch.empty(0)
    return all_embeddings, all_labels

class InferenceSetup():
    # For the dataloader that has returns x, target, is_query
    # To separate the query and gallery datasets
    def __init__(self):
        self.query_embeddings = []
        self.query_labels = []
        self.gallery_embeddings = []
        self.gallery_labels = []

    def append_to_query(self, embedding, label):
        # Ensure target[i] is a 1D tensor for consistency
        label = label if label.dim() > 0 else label.unsqueeze(0)
        self.query_embeddings.append(embedding)
        self.query_labels.append(label)
    
    def append_to_gallery(self, embedding, label):
        # Ensure target[i] is a 1D tensor for consistency
        label = label if label.dim() > 0 else label.unsqueeze(0)
        self.gallery_embeddings.append(embedding)
        self.gallery_labels.append(label)

    def concat_query(self):
        self.query_embeddings = torch.cat(self.query_embeddings)
        self.query_labels = torch.cat(self.query_labels)
        return self.query_embeddings, self.query_labels

    def concat_gallery(self):
        self.gallery_embeddings = torch.cat(self.gallery_embeddings)
        self.gallery_labels = torch.cat(self.gallery_labels)
        return self.gallery_embeddings, self.gallery_labels
    
class KnnClassifier:
    """
    Predict query label as k labels of nearest matches in database. If there is tie at given k,
    prediction from k-1 is used. Input is similarity matrix with `n_query` x `n_database` shape.


    Args:
        k: use k nearest in database for the majority voting.
        database_labels: list of labels in database. If provided, decode predictions to database
        (e.g. string) labels.
    Returns:
        1D array with length `n_query` of database labels (col index of the similarity matrix).
    """

    def __init__(self, k: int = 1, database_labels: np.array = None):
        self.k = k
        self.database_labels = database_labels

    def __call__(self, similarity):
        similarity = torch.tensor(similarity, dtype=float)
        scores, idx = similarity.topk(k=self.k, dim=1)
        pred = self.aggregate(idx)[:, self.k - 1]

        if self.database_labels is not None:
            print(len(pred))
            print(len(self.database_labels))
            pred = self.database_labels[pred]
        return pred

    def aggregate(self, predictions):
        """
        Aggregates array of nearest neighbours to single prediction for each k.
        If there is tie at given k, prediction from k-1 is used.

        Args:
            array of with shape [n_query, k] of nearest neighbours.
        Returns:
            array with predictions [n_query, k]. Column dimensions are predictions for [k=1,...,k=k]
        """

        results = defaultdict(list)
        for k in range(1, predictions.shape[1] + 1):
            for row in predictions[:, :k]:
                vals, counts = np.unique(row, return_counts=True)
                best = vals[np.argmax(counts)]

                counts_sorted = sorted(counts)
                if (len(counts_sorted)) > 1 and (counts_sorted[0] == counts_sorted[1]):
                    best = None
                results[k].append(best)

        results = pd.DataFrame(results).T.fillna(method="ffill").T
        return results.values
    
class KnnMatcher:
    """
    Find nearest match to query in existing database of features.
    Combines CosineSimilarity and KnnClassifier.
    """

    def __init__(self, database, k=1):
        self.similarity = CosineSimilarity()
        self.database = database
        self.classifier = KnnClassifier(
            database_labels=self.database.labels_string, k=k
        )

    def __call__(self, query):
        if isinstance(query, list):
            query = np.concatenate(query)

        if not isinstance(query, np.ndarray):
            raise ValueError("Query should be array or list of features.")

        sim_matrix = self.similarity(query, self.database.features)["cosine"]
        return self.classifier(sim_matrix)



def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if False(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())                 #同类索引
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())                 #非同类索引

    # dist_ap means distance(anchor, positive)
    # both dist_ap and relative_p_inds with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # dist_an means distance(anchor, negative)
    # both dist_an and relative_n_inds with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

def batch_hard_mining_embeddings(embeddings, labels):
    """
    Perform batch hard mining to find the hardest positive and hardest negative for each anchor in the batch.
    
    Args:
        embeddings (torch.Tensor): Embedding matrix of shape (N, D) where N is the batch size and D is the embedding dimension.
        labels (torch.Tensor): Labels corresponding to the embeddings of shape (N,).
    
    Returns:
        torch.Tensor: Triplet loss for the batch.
    """
    # Calculate pairwise distance matrix
    distance_matrix = euclidean_dist(embeddings, embeddings)
    
    # Initialize lists to hold triplet losses
    triplet_losses = []
    
    for i in range(len(labels)):
        # Get the anchor label
        anchor_label = labels[i]

        # Get the distances for the current anchor
        distances = distance_matrix[i]

        # Positive mask: True for examples of the same class
        positive_mask = (labels == anchor_label).float()

        # Negative mask: True for examples of different classes
        negative_mask = (labels != anchor_label).float()

        # Mask out the anchor itself in the positive mask
        positive_mask[i] = 0

        # Find the hardest positive: maximum distance to any positive example
        hardest_positive = torch.max(distances * positive_mask)

        # Find the hardest negative: minimum distance to any negative example
        hardest_negative = torch.min(distances * negative_mask + (1 - negative_mask) * 1e12)  # 1e12 is a large value to ignore zeros

        # Calculate Triplet Loss
        triplet_loss = F.relu(hardest_positive - hardest_negative + margin)
        triplet_losses.append(triplet_loss)

    # Average the triplet losses
    triplet_loss = torch.mean(torch.stack(triplet_losses))
    
    return triplet_loss

def batch_hard_mining_by_label(labels, num_classes):
    """
    Perform batch hard mining based on labels to find the hardest positive
    and hardest negative for each anchor in the batch.
    
    Args:
        labels (torch.Tensor): Labels corresponding to the embeddings of shape (N,).
        num_classes (int): Total number of classes.
    
    Returns:
        list of tuples: Each tuple contains (anchor_index, hardest_positive_index, hardest_negative_index)
    """
    batch_size = labels.size(0)
    triplets = []

    for i in range(batch_size):
        anchor_label = labels[i].item()

        # Get indices of all samples with the same label (positive samples)
        positive_indices = (labels == anchor_label).nonzero(as_tuple=False).view(-1)
        positive_indices = positive_indices[positive_indices != i]  # Exclude the anchor itself
        
        # Hardest positive: any sample with the same label (could choose randomly or the first one)
        hardest_positive_index = positive_indices[0] if positive_indices.numel() > 0 else i

        # Find the label farthest from the anchor label
        label_diffs = torch.abs(labels - anchor_label)
        max_diff = label_diffs.max().item()

        # Get indices of samples with the maximum label difference
        negative_indices = (label_diffs == max_diff).nonzero(as_tuple=False).view(-1)
        
        # Hardest negative: any sample with the farthest label (could choose randomly or the first one)
        hardest_negative_index = negative_indices[0] if negative_indices.numel() > 0 else i

        triplets.append((i, hardest_positive_index.item(), hardest_negative_index.item()))

    return triplets



class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
    

class TripletLoss_wildlife(nn.Module):
    """
    Wraps Pytorch Metric Learning TripletMarginLoss.

    Mining is one of: 'all', 'hard', 'semihard'
    Distance is one of: 'cosine', 'l2', 'l2_squared'
    """

    def __init__(self, margin=0.2, mining="seminard", distance="l2_squared"):
        super().__init__()
        if distance == "cosine":
            distance = distances.CosineSimilarity()
        elif distance == "l2":
            distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
        elif distance == "l2_squared":
            distance = distances.LpDistance(normalize_embeddings=True, p=2, power=2)
        else:
            raise ValueError(f"Invalid distance: {distance}")

        self.loss = losses.TripletMarginLoss(distance=distance, margin=margin)
        self.miner = miners.TripletMarginMiner(
            distance=distance, type_of_triplets=mining, margin=margin
        )

    def forward(self, embeddings, y):
        indices_tuple = self.miner(embeddings, y)
        return self.loss(embeddings, y, indices_tuple)