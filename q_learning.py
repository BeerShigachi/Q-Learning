import gym
import numpy as np
import pprint

env = gym.make("MountainCar-v0")
observation = env.reset()

LEARNING_ALPHA = 0.1
DISCOUNT = 0.95
DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)
EPISODES = 100000

SHOW_RATE = 100

discreate_obs_window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))


def get_discrete_state(obs):
    res = (obs - env.observation_space.low) / discreate_obs_window_size
    return tuple(res.astype(np.int_))


discrete_observation = get_discrete_state(observation)

for ep in range(EPISODES):

    if ep % SHOW_RATE == 0:
        print(ep)
        env.render()

    action = np.argmax(q_table[discrete_observation])  # agent here
    observation, reward, done, info = env.step(action)
    new_observation = get_discrete_state(observation)
    if done:
        observation = env.reset()
        if new_observation[0] >= env.goal_position:
            q_table[discrete_observation + (action,)] = 0
            continue
        else:
            print(f"done on episode {ep}!")
            break
    max_next_q_val = np.max(q_table[new_observation])
    current_q_val = q_table[discrete_observation + (action,)]

    new_q_val = (1 - LEARNING_ALPHA) * current_q_val + LEARNING_ALPHA * (reward + DISCOUNT * max_next_q_val)
    q_table[discrete_observation + (action,)] = new_q_val

    discrete_observation = new_observation

env.close()
