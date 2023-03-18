import numpy as np
import time
import gym

env = gym.make('MountainCar-v0')

observation = env.reset()  # 状态

for t in range(500):  #
    action = np.random.choice([0, 1, 2])
    observation, reward, done, info = env.step(action)
    print(observation.shape)
    env.render()
    time.sleep(0.02)
env.close()
