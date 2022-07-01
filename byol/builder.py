# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.nn as nn


class BYOL(nn.Module):
    """
    Build a BYOL model.
    """
    def __init__(self, base_encoder, hidden_dim=2048, pred_dim=256, momentum=0.99):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(BYOL, self).__init__()

        self.momentum = momentum
        # create the online encoder
        self.online_encoder = base_encoder

        # build a 3-layer projector
        self.online_encoder = replace_fc_with_mlp(self.online_encoder, hidden_dim, pred_dim)

        # create the target encoder
        self.target_encoder = copy.deepcopy(self.online_encoder)
        # freeze target encoder
        for target_weight in self.target_encoder.parameters():
            target_weight.requires_grad = False

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(pred_dim, hidden_dim, bias=False),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(hidden_dim, pred_dim))  # output layer

    @torch.no_grad()
    def _update_moving_average(self):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = target_params.data * self.momentum + online_params.data * (1 - self.momentum)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        # compute features for one view
        online_z1 = self.online_encoder(x1)  # NxC
        online_z2 = self.online_encoder(x2)  # NxC

        p1 = self.predictor(online_z1)  # NxC
        p2 = self.predictor(online_z2)  # NxC

        with torch.no_grad():
            self._update_moving_average()
            target_z1 = self.target_encoder(x1)  # NxC
            target_z2 = self.target_encoder(x2)  # NxC

        return p1, p2, target_z1.detach(), target_z2.detach()

def replace_fc_with_mlp(encoder, hidden_dim, pred_dim):
    module_names = [item[0] for item in encoder._modules.items()]
    if '_fc' in module_names:
        prev_dim = encoder._fc.weight.shape[1]
        encoder._fc = nn.Sequential(nn.Linear(prev_dim, hidden_dim, bias=False),
                                            nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(inplace=True),  # hidden layer
                                            nn.Linear(hidden_dim, pred_dim))  # output layer
    elif 'fc' in module_names:
        prev_dim = encoder.fc.weight.shape[1]
        encoder.fc = nn.Sequential(nn.Linear(prev_dim, hidden_dim, bias=False),
                                            nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(inplace=True),  # hidden layer
                                            nn.Linear(hidden_dim, pred_dim))  # output layer
    
    return encoder