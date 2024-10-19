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


"""
Base code https://github.com/owahltinez/triplet-loss-animal-reid/blob/main/mean_average_precision.py
Below:
"""

import math
import functools
import multiprocessing

import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback


def _hitcount_embedding_open_set(
    i: int,
    y_true: torch.Tensor = None,
    y_pred: torch.Tensor = None,
    top_k: tuple[int] = None,
) -> dict[int, int]:
  y1 = y_true[i]
  diff = [j for j, y2 in enumerate(y_true) if y1 != y2]
  same = [j for j, y2 in enumerate(y_true) if i != j and y1 == y2]

  if len(diff) > 0 and len(same) > 0:
    dist_diff = [(np.linalg.norm(y_pred[i] - y_pred[j]), 0) for j in diff]
    dist_same = [(np.linalg.norm(y_pred[i] - y_pred[j]), 1) for j in same]
    dist_combined = np.random.permutation(dist_diff + dist_same)
    hits_sorted = [h for _, h in sorted(dist_combined, key=lambda t: t[0])]
    return {k: 1 if sum(hits_sorted[:k]) > 0 else 0 for k in top_k}

  else:
    return {k: 0 for k in top_k}


def _hitcount_embedding_closed_set(
    class_label: str,
    train_embeddings: torch.Tensor = None,
    test_embeddings: torch.Tensor = None,
    train_labels: list[str] = None,
    test_labels: list[str] = None,
    top_k: tuple[int] = None,
    vote_count: int = 1,
) -> dict[int, int]:
  # Go through all embeddings in the test set.
  votes = {label: 0 for label in set(train_labels)}
  test_embeddings = [v for v, label in zip(test_embeddings, test_labels) if label == class_label]
  for v_test in test_embeddings:
    # Compute the distance between each test embedding and all other train embeddings.
    train_iter = zip(train_embeddings, train_labels)
    distances = [(label, np.linalg.norm(v_test - v_train)) for v_train, label in train_iter]
    distances = list(sorted(np.random.permutation(distances), key=lambda tup: tup[1]))

    # Add a vote for each embedding in the nearest <vote_count> distances.
    for label, _ in distances[:vote_count]:
      votes[label] += 1

  # Sort the votes in descending order and compute number of hits in the top-k items.
  votes_sorted = list(sorted(votes.items(), key=lambda tup: -tup[1]))
  hits_sorted = [int(label == class_label) for label, _ in votes_sorted]
  return {k: 1 if sum(hits_sorted[:k]) > 0 else 0 for k in top_k}


def _score_embeddings_parallel(score_func, map_iter, **kwargs) -> dict[str, float]:
  top_k = kwargs.pop("top_k")

  total = 0
  hits = {k: 0 for k in top_k}

  map_func = functools.partial(score_func, **kwargs, top_k=top_k)
  with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    for res in pool.imap_unordered(map_func, map_iter):
      total += 1
      for k, x in res.items():
        hits[k] += x

  return {f"mAP@{k}": hits[k] / total for k in top_k}


def evaluate_model_open_set(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device = torch.device('cpu'),
    top_k: tuple[int] = (1,),
) -> dict[str, float]:
  # Separate labels and data so we can feed the labels back to the score function.
  images, labels = tuple(zip(*test_data))
  y_true = torch.cat(labels, dim=0)

  # Compute embeddings and score based on distances.
  y_pred = model(torch.cat(images, dim=0))
  assert y_true.shape[0] == y_pred.shape[0], f"{len(y_true)} != {len(y_pred)}"
  return _score_embeddings_parallel(
      _hitcount_embedding_open_set,
      range(y_true.shape[0]),
      y_true=y_true,
      y_pred=y_pred,
      top_k=top_k,
  )


def evaluate_model_closed_set(
    model: torch.nn.Module,
    train_data: DataLoader,
    test_data: DataLoader,
    device: torch.device = torch.device('cpu'),
    top_k: tuple[int] = (1,),
    vote_count: int = 5,
) -> dict[str, float]:
  # Compute embeddings for all images in train set.
  train_images, train_classes = [torch.cat(x, 0) for x in zip(*train_data)]
  train_embeddings = model(train_images.to(device))
  train_labels = [train_data.class_names[k] for k in train_classes]

  # Compute embeddings for all images in test set.
  test_images, test_classes = [torch.cat(x, 0) for x in zip(*test_data)]
  test_embeddings = model(test_images.to(device))
  test_labels = [test_data.class_names[k] for k in test_classes]

  # Train dataset is triplet-safe, which leads to many duplicate elements.
  train_embeddings_unique = {}
  for embedding, label, image in zip(train_embeddings, train_labels, train_images):
    embedding_string = embedding.tobytes()
    if embedding_string not in train_embeddings_unique:
      train_embeddings_unique[embedding_string] = (embedding, label, image)
  train_embeddings, train_labels, train_images = list(zip(*train_embeddings_unique.values()))

  return _score_embeddings_parallel(
      _hitcount_embedding_closed_set,
      test_data.class_names,
      train_embeddings=train_embeddings,
      test_embeddings=test_embeddings,
      train_labels=train_labels,
      test_labels=test_labels,
      top_k=top_k,
      vote_count=vote_count,
  )


class MeanAveragePrecisionCallback(Callback):

  def __init__(self, top_k: tuple[int], dataset: DataLoader, sample_size: int = None):
    super().__init__()
    self.top_k = top_k
    self.dataset = dataset
    self.sample_size = sample_size

    # If we are using only a sample of the dataset, make it repeat and infer batch size.
    if self.sample_size is not None and self.sample_size < len(self.dataset):
        self._batch_size = next(iter(self.dataset))[0].shape[0]
        indices = torch.randperm(len(self.dataset))[:self.sample_size]
        self.dataset = torch.utils.data.Subset(self.dataset, indices)


  def on_epoch_end(self, epoch, logs=None):
    data = self.dataset

    # If we only need a sample, take as many batches as needed.
    if self.sample_size is not None and self.sample_size < len(self.dataset):
      data = DataLoader(self.dataset, batch_size=self._batch_size)

    # Evaluate the model using our scoring function.
    logs.update(evaluate_model_open_set(self.model, data, self.top_k))