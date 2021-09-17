import sys
import os
import gym
import numpy as np
import matplotlib.pyplot as plt


SAVE_DIR = 'tables'

try:
    os.mkdir(SAVE_DIR)
except FileExistsError:
    pass
except Exception as e:
    print("unknown error %s" % e)

env = gym.make("MountainCar-v0")

LEARNING_ALPHA = 0.1
DISCOUNT = 0.95
DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)
EPISODES = 20000
SHOW_RATE = 500
discrete_obs_window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE


try:
    q_table = np.load(sys.argv[1], allow_pickle=False)
    print("use %s" % sys.argv[1])
except FileNotFoundError:
    sys.exit("Invalid file path. '%s'" % sys.argv[1])
except IndexError:
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))

epsilon = 0.5
EPSILON_DECAY_START = 1  # probably much higher starting episodes
EPSILON_DECAY_END = EPISODES // 2
epsilon_decay_val = epsilon / (EPSILON_DECAY_END - EPSILON_DECAY_START)

episode_rewards = []
aggr_episode_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state(obs):
    res = (obs - env.observation_space.low) / discrete_obs_window_size
    return tuple(res.astype(np.int_))


for ep in range(EPISODES):
    ep_reward = 0
    render = False
    if ep % SHOW_RATE == 0:
        print(ep)
        render = True

    discrete_observation = get_discrete_state(env.reset())
    done = False

    while not done:
        action = np.random.randint(0, env.action_space.n)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_observation])  # agent here
        observation, reward, done, info = env.step(action)
        ep_reward += reward
        new_observation = get_discrete_state(observation)
        if render:
            env.render()
        if not done:
            max_next_q_val = np.max(q_table[new_observation])
            current_q_val = q_table[discrete_observation + (action,)]

            new_q_val = (1 - LEARNING_ALPHA) * current_q_val + LEARNING_ALPHA * (reward + DISCOUNT * max_next_q_val)

            q_table[discrete_observation + (action,)] = new_q_val

        elif observation[0] >= env.goal_position:
            print(f"done! on ep{ep}", q_table[discrete_observation + (action,)], action)
            q_table[discrete_observation + (action,)] = 0

        discrete_observation = new_observation

    if EPSILON_DECAY_END >= ep >= EPSILON_DECAY_START:
        epsilon -= epsilon_decay_val

    episode_rewards.append(ep_reward)

    if not ep % 100 and ep > 0:
        np.save("%s/%d.npy" % (SAVE_DIR, ep), q_table, allow_pickle=False)

    if not ep % SHOW_RATE:  ## equivalent to if episode%Show_rate == 0:
        average_reward = sum(episode_rewards[-SHOW_RATE:]) / len(episode_rewards[-SHOW_RATE:])
        aggr_episode_rewards['ep'].append(ep)
        aggr_episode_rewards['avg'].append(average_reward)
        aggr_episode_rewards['min'].append(min(episode_rewards[-SHOW_RATE:]))
        aggr_episode_rewards['max'].append(max(episode_rewards[-SHOW_RATE:]))

        print('episode: ', aggr_episode_rewards['ep'], 'avg: ', average_reward, 'min: ', min(episode_rewards[-SHOW_RATE:]), 'max: ', max(episode_rewards[-SHOW_RATE:]))
env.close()

plt.plot(aggr_episode_rewards['ep'], aggr_episode_rewards['avg'], label='avg')
plt.plot(aggr_episode_rewards['ep'], aggr_episode_rewards['min'], label='min')
plt.plot(aggr_episode_rewards['ep'], aggr_episode_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
