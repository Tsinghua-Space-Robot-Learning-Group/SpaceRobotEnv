from multiprocessing.context import BaseContext
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
PATH = os.getcwd()
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, model_name : str , fc1_dims = 256,\
     fc2_dims = 256, check_point_base_dir = 'Learning_algorithm/Torch/PPO/models/') -> None:
        super(CriticNetwork, self).__init__()

        check_point_base_dir = os.path.join( PATH , check_point_base_dir )
        self.check_point_file = os.path.join(check_point_base_dir, model_name)
        self.critic  = nn.Sequential(
            nn.Linear(*input_dims , fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims , fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims , 1),

        )
        self.optimiser = optim.Adam(self.parameters(), lr = alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.check_point_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.check_point_file))