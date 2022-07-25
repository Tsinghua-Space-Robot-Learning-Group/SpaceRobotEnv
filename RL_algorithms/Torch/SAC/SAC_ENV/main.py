from turtle import shape
import gym
import numpy as np
from agent import AgentController

from utils import plot_learning_curve
import gym

import SpaceRobotEnv
import numpy as np



if __name__ == '__main__':
    env = gym.make("SpaceRobotState-v0")
    N = 20
    batch_size = 5 
    n_epochs = 4 
    alpha = 0.0003
    action_space = env.action_space.shape[0]
    obs_shape = env.observation_space["observation"].shape
    agent = AgentController(model_name_critic1= "space_robot_critic1.pt",
                    model_name_critic2= "space_robot_critic2.pt",
                    model_name_value= "space_robot_value.pt",
                    model_name_target_value="space_robot_target_value.pt",
                    model_name_actor="space_robot_actor.pt",
                    input_shape = env.observation_space["observation"].shape,
                    env = env)


    n_iter = 300
    figure_file = 'RL_algorithms/Torch/SAC/plots/space_robot_performance.png'
    best_score = env.reward_range[0]
    score_history = []
    n_steps = 0
    learn_iters = 0
    avg_score = 0

    for i in range(n_iter):
        obs = env.reset()
        observation = obs["observation"].astype(np.float32)

        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = observation_["observation"].astype(np.float32)

            n_steps+=1
            score += reward
            agent.store_memory(observation,  action, reward, observation_, done)
            agent.learn()
            learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score>best_score:
            best_score= avg_score
            agent.save_models()
        print('episode', i , 'score %.1f', score, 'avg_score %.1f' %avg_score,
        'time_steps',n_steps, 'learning_steps', learn_iters)

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    env.close()
