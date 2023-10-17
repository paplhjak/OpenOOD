from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class LikelihoodEstimatorPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
            output = net(data)
            conf, pred = torch.max(output, dim=1)
            smax = torch.softmax(output, dim=1)
            smax = smax + 10**-10
            conf = -(smax[:,0]/smax[:,1])
            return pred, conf
