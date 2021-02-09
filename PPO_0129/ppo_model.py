import torch
import torch.nn as nn
import torch.nn.functional as func


class PPOModel(nn.Module):

    def __init__(self, observation_dim, action_dim):
        super(PPOModel, self).__init__()
        self._fc_base_0 = nn.Linear(observation_dim, 128)
        # vnet
        self._vnet_fc_0 = nn.Linear(128, 128)
        self._vnet_fc_1 = nn.Linear(128, 64)
        self._vnet_fc_2 = nn.Linear(64, 1)
        # policy net
        self._pnet_fc_0 = nn.Linear(128, 128)
        self._pnet_fc_1 = nn.Linear(128, 64)
        self._pnet_fc_mu = nn.Linear(64, action_dim)
        # self._pnet_fc_sigma = nn.Linear(64, action_dim)

    def forward(self, x):
        base = torch.relu(self._fc_base_0(x))
        v0 = torch.relu(self._vnet_fc_0(base))
        v1 = torch.relu(self._vnet_fc_1(v0))
        v = self._vnet_fc_2(v1)

        p0 = torch.relu(self._pnet_fc_0(base))
        p1 = torch.relu(self._pnet_fc_1(p0))
        p_mu = torch.tanh(self._pnet_fc_mu(p1))
        return v, p_mu

