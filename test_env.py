import gym

import SpaceRobotEnv
import numpy as np

env = gym.make("SpaceRobotReorientation-v0")

dim_u = env.action_space.shape[0]
print(dim_u)
dim_o = env.observation_space["observation"].shape[0]
print(dim_o)


observation = env.reset()
max_action = env.action_space.high
print("max_action:", max_action)
print("min_action", env.action_space.low)
for e_step in range(20):
    observation = env.reset()
    for i_step in range(50):
        env.render()
        action = np.random.uniform(low=-1.0, high=1.0, size=(dim_u,))
        observation, reward, done, info = env.step(max_action * action)

env.close()
