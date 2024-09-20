# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F


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