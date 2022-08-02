from multiprocessing.context import BaseContext
import os
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
PATH = os.getcwd()

class ActorNetwork(nn.Module):
    
    def __init__(self, max_actions, n_actions, input_dims, alpha,  model_name : str, 
            fc1_dims=256, fc2_dims=256, check_point_base_dir = 'RL_algorithms/Torch/PPO/Continious/PPO_Two_heads/models'):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.max_actions = max_actions
       
        
        check_point_base_dir = os.path.join( PATH , check_point_base_dir )
        self.checkpoint_file = os.path.join(check_point_base_dir, model_name)
        self.base_model = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
        )
        fc = [nn.Linear(fc2_dims, 2*n_actions)]
        self.fc = nn.Sequential(*fc)       
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.base_model(state)
        x = self.fc(x)
        mean, std = T.chunk(x, chunks=2, dim=-1)
        mean, std = self.max_actions * T.tanh(mean), F.softplus(std)
        return mean, std

    def get_logprob(self, state, action):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(axis=-1) 
        return log_prob

 
       
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
