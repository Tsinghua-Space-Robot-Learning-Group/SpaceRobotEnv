from turtle import shape
import gym
import numpy as np
from agent import Agent

from utils import plot_learning_curve

from SpaceRobotEnv.envs import SpaceRobotImage

import numpy as np



if __name__ == '__main__':
    # env = SpaceRobotImage()
    env = SpaceRobotImage()
    #N = 20
    N = 20
    batch_size = 5 
    n_epochs = 4 
    alpha = 0.0003
    action_space = env.action_space.shape[0]    
    agent = Agent(  n_actions = action_space, 
                    batch_size=batch_size, 
                    alpha = alpha,
                    n_epoch = n_epochs, 
                    model_name_actor = "space_robot_actor.pt",
                    model_name_critic = "space_robot_critic.pt")
    n_iter = 300
    figure_file = 'RL_algorithms/Torch/PPOImage/plots/space_robot_performance.png'
    best_score = env.reward_range[0]
    score_history = []
    n_steps = 0
    learn_iters = 0
    avg_score = 0

    for i in range(n_iter):
        obs = env.reset()
        observation = obs["rawimage"].reshape(3, 64, 64)
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward

            agent.remember(observation,  action, prob, val, reward, done)
            #steps before we begin learning 20
            if n_steps % N ==0:
                agent.learn()
                learn_iters += 1
            observation = observation_["rawimage"].reshape(3, 64, 64)
           
        print("done")
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score>best_score:
            best_score= avg_score
            agent.save_models()
        print('episode', i , 'score %.1f',score,  'avg_score %.1f' %avg_score,
        'time_steps',n_steps, 'learning_steps', learn_iters)

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history,figure_file)
    env.close()
