import torch.nn.functional as F
import torch.optim as optim
import gym
import pandas as pd
from model.agent import Agent


OUTPUT_GRAPH = False
MAX_EPISODE = 500000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 2000  # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic


# action有两个，即向左或向右移动小车
# state是四维

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

agent = Agent(N_F, N_A, GAMMA, LR_A, LR_C)

res = []
running_reward = None
show = False
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER:
            env.render()

        a = agent.choose_action(s, show)
        s_, r, done, info = env.step(a.item())

        # s_: shape:
        if done:
            r = -20
        track_r.append(r)

        agent.learn(s, a, r, s_)  # state, action, reward, s_next
        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if running_reward is None:
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.97 + ep_rs_sum * 0.03
            if running_reward > DISPLAY_REWARD_THRESHOLD or running_reward <= -20:
                RENDER = True  # rendering
            if i_episode % 10 == 0:
                print("episode:", i_episode, "  reward:", int(running_reward))
            res.append([i_episode, running_reward])

            break

pd.DataFrame(res, columns=['episode', 'ac_reward']).to_csv('../ac_reward.csv')
