import imp
from multiprocessing.context import BaseContext
import os
import copy
from tqdm import tqdm
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from actor import ActorNetwork
from critic import CriticNetwork 
from memory import PPOMemory


PATH = os.getcwd()
# MODEL_XML_PATH = os.path.join(
#     PATH, "SpaceRobotEnv", "assets", "spacerobot", "spacerobot_image.xml"
# )

class Agent:
    def __init__(self, n_actions,   model_name_actor : str, model_name_critic : str, \
                gamma = 0.99, alpha = 0.0003, gae_lambda = 0.95,  \
                policy_clip = 0.1, n_epoch = 10,  batch_size = 64):
        '''
        parameter 
            arguments:
                - model_name_actor : model name for actor to be used in model savind directory
                - model_name_critic :model name for critic to be used in model savind directory
        '''
        #self, n_actions, gae_lamda = 0.95, gamma = 0.99, alpha = 0 .0003, policy_clip = 0.2, batch_size = 64, N = 2048 , n_epoch = 10
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epoch = n_epoch

        self.actor = ActorNetwork(n_actions, alpha, model_name = model_name_actor)
        self.critic = CriticNetwork(alpha, model_name  = model_name_critic)
        self.memory_handler = PPOMemory( batch_size )

    def remember(self, state, action, probs, vals, reward, done):
        self.memory_handler.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("Saving models now")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_model(self):
        print("Load model")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def play_optimal(self, observation):
        with T.no_grad():
            state = T.tensor([observation], dtype=T.float).to(self.actor.device)
            dist = self.actor(state)
            # action shoulnt be sampe it should be arg max
            action = dist.sample()
            action =T.squeeze(action).item()
            return action

    def choose_action(self, observation):
        observation = np.array(observation)
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value  = self.critic(state)

        action = dist.sample()

        # this is equivalent to the reinforce algorithm of probablity distribition
        probs = T.squeeze(dist.log_prob(action)).item() 

        action =T.squeeze(action).item()
        value =T.squeeze(value).item()

        return action, probs , value

    def learn(self):
        for _ in range(self.n_epoch):

            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory_handler.generate_batches()

            values = vals_arr.copy()
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 0.95
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimiser.zero_grad()
                # print("total loss", total_loss.item())
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimiser.step()

        self.memory_handler.clear_memory()               
