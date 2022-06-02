import SpaceRobotEnv
import numpy as np

env = SpaceRobotEnv.SpaceRobotState()

dimu = env.action_space.shape
print(dimu)
dimo = env.observation_space['observation'].shape
print(dimo)


observation = env.reset()
max_action = env.action_space.high
print('max_action:',max_action)
print('mmin_action',env.action_space.low)
for e_step in range(20):
    observation = env.reset()
    for i_step in range(50):
        env.render()
        # print(observation)
        # action = env.action_space.sample()
        action = np.random.uniform(low=-1.0, high=1.0, size=(dimu,))
        observation, reward, done, info = env.step(max_action * action)

env.close()