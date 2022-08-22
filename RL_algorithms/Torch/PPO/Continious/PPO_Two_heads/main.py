from turtle import shape
import gym
import numpy as np
from agent import Agent

from utils import plot_learning_curve
import gym

import SpaceRobotEnv
import numpy as np



if __name__ == '__main__':
    env = gym.make("SpaceRobotState-v0")
    N = 30
    batch_size = 16
    n_epochs = 3
    alpha = 0.0003
    action_space = env.action_space.shape[0]
    obs_shape = env.observation_space["observation"].shape
    env_max_action =  float(env.action_space.high[0])
    
    agent = Agent(  env_max_action = env_max_action,
                    n_actions = action_space, 
                    batch_size = batch_size, 
                    alpha = alpha,
                    n_epoch = n_epochs, 
                    input_dims = obs_shape,
                    model_name_actor = "space_robot_actor.pt",
                    model_name_critic = "space_robot_critic.pt")
    n_iter = 300
    figure_file = 'RL_algorithms/Torch/PPO/Continious/PPO_Two_heads/plots/space_robot_performance.png'
    best_score = env.reward_range[0]
    score_history = []
    n_steps = 0
    learn_iters = 0
    avg_score = 0

    for i in range(n_iter):
        obs = env.reset()
        observation = obs["observation"]
      
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            
            action = action.detach().cpu().numpy().flatten()
            action = action.clip(env.action_space.low, env.action_space.high)

            action_logprob = prob.detach().cpu().numpy().flatten()
            val = val.detach().cpu().numpy().flatten()
            
            observation_, reward, done, info = env.step(action)
            n_steps+=1
            score += reward

            agent.remember(observation, action, action_logprob, val, reward, done)
            #steps before we begin learning 20
            if n_steps % N ==0:
                agent.learn()
                learn_iters += 1
            observation = observation_["observation"]
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score>best_score:
            best_score= avg_score
            agent.save_models()
        print('episode', i , 'score %.1f', 'avg_score %.1f' %avg_score,
        'time_steps',n_steps, 'learning_steps', learn_iters)

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history,figure_file)
    env.close()
