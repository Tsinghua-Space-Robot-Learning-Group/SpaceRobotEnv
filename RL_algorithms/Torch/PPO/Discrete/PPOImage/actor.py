from multiprocessing.context import BaseContext
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
PATH = os.getcwd()

class ActorNetwork(nn.Module):
    
    def __init__(self, n_actions, alpha,  model_name : str, 
            check_point_base_dir = 'RL_algorithms/Torch/PPOImage/models'):
        super(ActorNetwork, self).__init__()
        
        check_point_base_dir = os.path.join( PATH , check_point_base_dir )
        self.checkpoint_file = os.path.join(check_point_base_dir, model_name)
        
        self.actor = nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=5,  stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2,2),

                nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=5,  stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2,2),

                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=5,  stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2,2),

                nn.Flatten(),
                nn.Linear(1024, 4096),
                nn.ReLU(),
                nn.Linear(4096, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions),

                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
