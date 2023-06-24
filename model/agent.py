import torch
from .a2c import ActorNet, CriticNet
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Agent(object):
    def __init__(self, n_features, n_actions, GAMMA, lr_a, lr_c):
        self.actor = ActorNet(n_features, n_actions)
        self.critic = CriticNet(n_features, n_actions)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.GAMMA = GAMMA

    def choose_action(self, state, show=False):
        state = torch.Tensor(state).reshape(1, -1)
        probs = self.actor(state)
        if show:
            print("probs: ", probs)
        m = Categorical(probs)
        action = m.sample()
        return action

    def learn(self, state, action, reward, s_next):
        # train critic
        state = torch.Tensor(state).reshape(1, -1)
        s_next = torch.Tensor(s_next).reshape(1, -1)

        v_next = self.critic(s_next)
        v = self.critic(state)
        td_error = reward + self.GAMMA * v_next - v
        loss = torch.sum(torch.square(td_error))
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()
        # train actor
        probs = self.actor(state)
        log_prob = torch.log(probs[0, action])
        exp_v = -torch.mean(log_prob * td_error.detach())
        self.optimizer_actor.zero_grad()
        exp_v.backward()
        self.optimizer_actor.step()