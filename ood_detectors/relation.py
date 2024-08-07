import torch
from typing import Dict

from ood_detectors.interface import OODDetector
#from ood_detectors.assets import knn_score
from math import ceil


def normalize(feat, nc=79490):
    with torch.no_grad():
        split = ceil(len(feat) / nc)
        for i in range(split):
            feat_ = feat[i * nc:(i + 1) * nc]
            feat[i * nc:(i + 1) *
                        nc] = feat_ / torch.sqrt((feat_ ** 2).sum(-1) + 1e-10).reshape(-1, 1)

    return feat


def kernel(feat, feat_t, prob, prob_t, split=2):
    """Kernel function (assume feature is normalized)
    """
    size = ceil(len(feat_t) / split)
    rel_full = []
    for i in range(split):
        feat_t_ = feat_t[i * size:(i + 1) * size]
        prob_t_ = prob_t[i * size:(i + 1) * size]

        with torch.no_grad():
            dot = torch.matmul(feat, feat_t_.transpose(1, 0))
            dot = torch.clamp(dot, min=0.)

            sim = torch.matmul(prob, prob_t_.transpose(1, 0))
            rel = dot * sim

        rel_full.append(rel)

    rel_full = torch.cat(rel_full, dim=-1)
    return rel_full


def get_relation(feat, feat_t, prob, prob_t, pow=8, chunk=50, thres=0.03):
    """Get relation values (top-k and summation)

    Args:
        feat (torch.Tensor [N,D]): features of the source data
        feat_t (torch.Tensor [N',D]): features of the target data
        prob (torch.Tensor [N,C]): probabilty vectors of the source data
        prob_t (torch.Tensor [N',C]): probabilty vectors of the target data
        pow (int): Temperature of kernel function
        chunk (int): batch size of kernel calculation (trade off between memory and speed)
        thres (float): cut off value for small relation graph edges. Defaults to 0.03.

    Returns:
        graph: statistics of relation graph
    """

    n = feat.shape[0]
    n_chunk = ceil(n / chunk)

    score = []
    for i in range(n_chunk):
        feat_ = feat[i * chunk:(i + 1) * chunk]
        prob_ = prob[i * chunk:(i + 1) * chunk]

        rel = kernel(feat_, feat_t, prob_, prob_t)

        mask = (rel.abs() > thres)
        rel_mask = mask * rel
        edge_sum = (rel_mask.sign() * (rel_mask.abs() ** pow)).sum(-1)
        
        score.append(edge_sum.cpu())
        print(len(score))
    score = torch.cat(score, dim=0)

    return score




class RelationOODDetector(OODDetector):

    def setup(self, args, train_model_outputs):

        logits_train = train_model_outputs['logits']
        feas_train = train_model_outputs['feas']
        train_labels = train_model_outputs['labels']
        feas_train = normalize(feas_train)
        self.prob_train = torch.softmax(logits_train, dim=1)
        self.feas_train = feas_train

    def infer(self, model_outputs):
        feas = model_outputs['feas']
        logits = model_outputs['logits']
        prob = torch.softmax(logits,dim=1)
        feas = normalize(feas)
        score = get_relation(feas, self.feas_train, prob, self.prob_train, pow=1)
        #print(self.scaled_feas_train,'scaled_feas_train')
        return score