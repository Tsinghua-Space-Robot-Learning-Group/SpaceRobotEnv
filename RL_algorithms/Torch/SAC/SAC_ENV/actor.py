import imp
from multiprocessing.context import BaseContext
import os
import numpy as np
import torch as T

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
PATH = os.getcwd()

class ActorNetwork(nn.Module):
    
    def __init__(self, n_actions, input_dims, max_actions , alpha,  model_name : str, 
            fc1_dims=256, fc2_dims=256, 
            check_point_base_dir = 'RL_algorithms/Torch/SAC/models'):

        super(ActorNetwork, self).__init__()
        self.max_actions = max_actions

        self.n_actions = n_actions
        self.reparam_noise = 1e-6
    
        check_point_base_dir = os.path.join( PATH , check_point_base_dir )
        self.checkpoint_file = os.path.join(check_point_base_dir, model_name)

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.mu =  nn.Linear(fc2_dims , self.n_actions )
        self.sigma = nn.Linear(fc2_dims , self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise , max=1)
        return mu ,sigma
        
    def sample_normal (self, state, reparamaterize = True):

        mu, sigma = self.forward(state)
        probabilities =  Normal(mu, sigma)
        if reparamaterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        action = T.tanh(actions) * T.tensor(self.max_actions).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim = True)
        return action , log_probs
       
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
