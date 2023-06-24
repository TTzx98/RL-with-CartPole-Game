import torch
from torch import nn


class ActorNet(nn.Module):
    def __init__(self, n_features, n_actions):
        super().__init__()
        self.linear1 = nn.Linear(n_features, 20)
        self.linear2 = nn.Linear(20, n_actions)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.softmax(self.linear2(x), dim=-1)
        return x


class CriticNet(nn.Module):
    def __init__(self, n_features, n_actions):
        super().__init__()
        self.linear1 = nn.Linear(n_features, 20)
        self.linear2 = nn.Linear(20, 40)
        self.linear3 = nn.Linear(40, 1)
        self.n_actions = n_actions

    def forward(self, state):
        s_a = torch.concat([state], dim=-1)
        out = torch.relu(self.linear1(s_a))
        out = torch.relu(self.linear2(out))
        out = self.linear3(out)
        return out


