import gym
import numpy as np

env = gym.make("MountainCar-v0")
observation = env.reset()

LEARNING_ALPHA = 0.1
DISCOUNT = 0.95
DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)

discreate_obs_window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))


def get_discrete_state(observation):
    res = (observation - env.observation_space.low) / discreate_obs_window_size
    return tuple(res.astype(np.int_))


discrete_observation = get_discrete_state(observation)
print(discrete_observation)

for _ in range(10000):
    env.render()
    action = 2  # agent here
    observation, reward, done, info = env.step(action)


    if done:
        observation = env.reset()
env.close()
