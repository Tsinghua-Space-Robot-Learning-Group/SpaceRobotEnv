import imp
from multiprocessing.context import BaseContext
import os
import copy
from tqdm import tqdm
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from actor import ActorNetwork
from critic import CriticNetwork 
from memory import PPOBuffer


PATH = os.getcwd()
# MODEL_XML_PATH = os.path.join(
#     PATH, "SpaceRobotEnv", "assets", "spacerobot", "spacerobot_image.xml"
# )

class Agent:
    def __init__(self, env_max_action, n_actions, input_dims,  model_name_actor : str, model_name_critic : str, \
                gamma = 0.99, alpha = 0.0003, gae_lambda = 0.95,  \
                policy_clip = 0.2, n_epoch = 3,  batch_size = 64):
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

        self.actor = ActorNetwork( env_max_action , n_actions, input_dims, alpha, model_name = model_name_actor)
        self.critic = CriticNetwork(input_dims, alpha, model_name  = model_name_critic)
        self.memory_handler = PPOBuffer( batch_size )

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

   
    def choose_action(self, state):
        # state = T.as_tensor(state, dtype=T.float, device=device)
        state = T.tensor([state], dtype=T.float).to(self.actor.device)
        
        mean, std = self.actor.forward(state)
       
        dist = Normal(mean, std)
      

        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(axis=-1) 
        value = self.critic(state)

        return action, action_logprob,  value

    # def choose_action(self, observation):
    #     with T.no_grad():
    #         observation = T.tensor([observation], dtype=T.float).to(self.actor.device)
    #         action , logp_a  = self.actor.sample_normal(observation)
    #         value = self.critic(observation)
    #         return action.numpy(),  logp_a.numpy(), value.numpy()
    def learn(self):
        for _ in range(self.n_epoch):

            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory_handler.generate_batches()

            values = vals_arr.copy()
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            # calculate advantage = sigma_t + (gamma * lamda) * sigma_t+1 + (gamma * lamda) ^ 2 * sigma_t+2.....
            # sigma_t = reward_t + gamma * Value(s_ t+1 ) - Value(s_t)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    
                    a_t += discount * (reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])

                    #   discount term gamma * gae_lamda (y*lamda)
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)

                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                new_probs = self.actor.get_logprob(states, actions)

                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)
               
                prob_ratio = T.exp(new_probs - old_probs)
                
                weighted_probs = advantage[batch] * prob_ratio


                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1 + self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5* critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimiser.zero_grad()
                # print("total loss", total_loss.item())
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimiser.step()

        self.memory_handler.clear_memory()               
