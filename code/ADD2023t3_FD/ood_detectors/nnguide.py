import torch
from typing import Dict

from ood_detectors.interface import OODDetector
from ood_detectors.assets import knn_score
import numpy as np
# normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
import torch.nn.functional as F
from copy import deepcopy
import faiss


# import fastdtw

def knn_score_with_angle(feas_train, feas, k=10, min=False):
    feas_train = feas_train.cpu()
    feas_train = deepcopy(np.array(feas_train))
    feas = deepcopy(np.array(feas))
    # faiss.normalize_L2(feas_train)
    # faiss.normalize_L2(feas)
    index = faiss.IndexFlatIP(feas_train.shape[-1])
    index.add(feas_train)
    D, I = index.search(feas, k)

    if min:
        scores = np.array(D.min(axis=1))
    else:
        scores = np.array(D.mean(axis=1))

    return scores


class NNGuideOODDetector(OODDetector):

    def setup(self, args, train_model_outputs):
        logits_train = train_model_outputs['logits']
        feas_train = train_model_outputs['feas']
        train_labels = train_model_outputs['labels']
        feas_train = F.normalize(feas_train, p=2, dim=-1)
        self.knn_k = 6

        confs_train = torch.logsumexp(logits_train, dim=1)
        self.scaled_feas_train = feas_train * confs_train[:, None]
        # self.scaled_feas_train = feas_train

    def infer(self, model_outputs):
        feas = model_outputs['feas']
        logits = model_outputs['logits']

        # Calculate cosine similarity
        feas = F.normalize(feas, p=2, dim=-1)
        confs = torch.logsumexp(logits, dim=1)

        # Use cosine similarity for KNN
        guidances = knn_score_with_angle(self.scaled_feas_train, feas, k=self.knn_k)
        scores = torch.from_numpy(guidances).to(confs.device) * confs
        return scores