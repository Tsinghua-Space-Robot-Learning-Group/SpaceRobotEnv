from multiprocessing.context import BaseContext
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
PATH = os.getcwd()
class CriticNetwork(nn.Module):
    def __init__(self, alpha, model_name : str ,\
      check_point_base_dir = 'RL_algorithms/Torch/PPOImage/models/') -> None:
        super(CriticNetwork, self).__init__()

        check_point_base_dir = os.path.join( PATH , check_point_base_dir )
        self.check_point_file = os.path.join(check_point_base_dir, model_name)
        self.critic  = nn.Sequential(
           nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=5,  stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2,2),

                # nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=5,  stride=1),
                # nn.ReLU(),
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.MaxPool2d(2,2),

                # nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=5,  stride=1),
                # nn.ReLU(),
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.MaxPool2d(2,2),

                nn.Flatten(),
                nn.Linear(28800, 512),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 1),

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