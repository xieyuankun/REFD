import torch
from typing import Dict

from ood_detectors.interface import OODDetector
import numpy as np
# normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
import torch.nn.functional as F
from copy import deepcopy



def NSD_with_angle(feas_train, feas, min=False):
    feas_train = feas_train.cpu().numpy()
    feas = feas.cpu().numpy()

    # Calculate cosine similarity

    cos_similarity = np.dot(feas, feas_train.T)
    print(feas.shape, 'feas.shape')
    print(feas_train.T.shape, 'feas_train.T.shape')
    print(cos_similarity.shape, 'cos_similarity.shape')
    if min:
        scores = np.array(cos_similarity.min(axis=1))
    else:
        scores = np.array(cos_similarity.mean(axis=1))
    return scores


class NSDOODDetector(OODDetector):

    def setup(self, args, train_model_outputs):
        logits_train = train_model_outputs['logits']
        feas_train = train_model_outputs['feas']
        train_labels = train_model_outputs['labels']
        feas_train = F.normalize(feas_train, p=2, dim=-1)
        confs_train = torch.logsumexp(logits_train, dim=1)
        self.scaled_feas_train = feas_train * confs_train[:, None]

    def infer(self, model_outputs):
        feas = model_outputs['feas']
        logits = model_outputs['logits']

        # Calculate cosine similarity
        feas = F.normalize(feas, p=2, dim=-1)
        confs = torch.logsumexp(logits, dim=1)

        guidances = NSD_with_angle(self.scaled_feas_train, feas)
        scores = torch.from_numpy(guidances).to(confs.device) * confs
        return scores