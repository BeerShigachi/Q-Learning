import gym
import numpy as np

env = gym.make("MountainCar-v0")

LEARNING_ALPHA = 0.1
DISCOUNT = 0.95
DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)
EPISODES = 5000
SHOW_RATE = 200
discreate_obs_window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))

epsilon = 0.5
EPSILON_DECAY_START = 1  # probably much higher starting episodes
EPSILON_DECAY_END = EPISODES // 2
epsilon_decay_val = epsilon / (EPSILON_DECAY_END - EPSILON_DECAY_START)


def get_discrete_state(obs):
    res = (obs - env.observation_space.low) / discreate_obs_window_size
    return tuple(res.astype(np.int_))


for ep in range(EPISODES):

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

env.close()
