# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
# import torch
# from ignite.metrics import Metric

# from data.datasets.eval_reid import eval_func
# from .re_ranking import re_ranking



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


# class R1_mAP(Metric):
#     def __init__(self, num_query, max_rank=50, feat_norm='yes', is_demo=False):
#         super(R1_mAP, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm
#         self.is_demo = is_demo

#     def reset(self):
#         self.feats = []
#         self.pids = []
#         self.camids = []
#         self.paths = []
#         self.score = []

#     def update(self, output):
#         feat, pid, camid, path = output
#         self.feats.append(feat)
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))
#         self.paths.extend(np.asarray(path))

#     def compute(self):
#         feats = torch.cat(self.feats, dim=0)
#         if self.feat_norm == 'yes':
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)
#         # query
#         qf = feats[:self.num_query]
#         q_pids = np.asarray(self.pids[:self.num_query])
#         q_camids = np.asarray(self.camids[:self.num_query])
#         q_paths = np.asarray(self.paths[:self.num_query])
#         # gallery
#         gf = feats[self.num_query:]
#         g_pids = np.asarray(self.pids[self.num_query:])
#         g_camids = np.asarray(self.camids[self.num_query:])
#         g_paths = np.asarray(self.paths[self.num_query:])

#         #-------------------无re-rank
#         # m, n = qf.shape[0], gf.shape[0]
#         # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#         #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#         # distmat.addmm_(1, -2, qf, gf.t())
#         # distmat = distmat.cpu().numpy()

#         #-------------------re-rank
#         print("Enter reranking")
#         distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)


#         if self.is_demo:
#             # PATH = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_paths, g_paths, is_demo=True)
#             # return PATH
#             return distmat, q_paths, g_paths

#         else:
#             cmc, mAP, PATH = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_paths, g_paths)
#             return cmc, mAP, PATH

# class R1_mAP_reranking(Metric):
#     def __init__(self, num_query, max_rank=50, feat_norm='yes'):
#         super(R1_mAP_reranking, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm

#     def reset(self):
#         self.feats = []
#         self.pids = []
#         self.camids = []

#     def update(self, output):
#         feat, pid, camid = output
#         self.feats.append(feat)
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))

#     def compute(self):
#         feats = torch.cat(self.feats, dim=0)
#         if self.feat_norm == 'yes':
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)

#         # query
#         qf = feats[:self.num_query]
#         q_pids = np.asarray(self.pids[:self.num_query])
#         q_camids = np.asarray(self.camids[:self.num_query])
#         # gallery
#         gf = feats[self.num_query:]
#         g_pids = np.asarray(self.pids[self.num_query:])
#         g_camids = np.asarray(self.camids[self.num_query:])
#         # m, n = qf.shape[0], gf.shape[0]
#         # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#         #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#         # distmat.addmm_(1, -2, qf, gf.t())
#         # distmat = distmat.cpu().numpy()
#         print("Enter reranking")
#         distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
#         cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

#         return cmc, mAP