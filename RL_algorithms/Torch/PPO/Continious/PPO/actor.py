from multiprocessing.context import BaseContext
import os
import numpy as np
import torch as T

import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
PATH = os.getcwd()

class ActorNetwork(nn.Module):
    
    def __init__(self, n_actions, input_dims, alpha,  model_name : str, 
            fc1_dims=256, fc2_dims=256, check_point_base_dir = 'RL_algorithms/Torch/PPO/Continious/PPO/models/'):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions

        log_std = -0.5 * np.ones(n_actions, dtype=np.float32)
        self.log_std = T.nn.Parameter(T.as_tensor(log_std))
        
        check_point_base_dir = os.path.join( PATH , check_point_base_dir )
        self.checkpoint_file = os.path.join(check_point_base_dir, model_name)

        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, obs, act = None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
        
    def _distribution(self, state):
        mu = self.actor(state)
        std = T.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    
       
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
