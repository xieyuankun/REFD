import torch
from typing import Dict
import numpy as np
from ood_detectors.interface import OODDetector
from ood_detectors.assets import knn_score
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

class KNNOODDetector(OODDetector):

    def setup(self, args, train_model_outputs):
        feas_train = train_model_outputs['feas']
        
        self.knn_k = 6

        self.feas_train = normalizer(feas_train)

    def infer(self, model_outputs: Dict):

        feas = model_outputs['feas']
        feas = normalizer(feas)
        scores = knn_score(self.feas_train, feas, k=self.knn_k, min=True)
        scores = torch.from_numpy(scores).to(feas.device)
        return scores