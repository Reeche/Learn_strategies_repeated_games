import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import seed

torch.manual_seed(seed)
criterion = nn.MSELoss()


class Actor(nn.Module):
    """
    Aim: takes the current state as input and learn to output an optimal strategy
    """
    def __init__(self, input_size_A, hidden_size_A, output_size_A):
        super(Actor, self).__init__()
        self.saved_actions = []

        self.map1 = nn.Linear(input_size_A, hidden_size_A)
        self.map2 = nn.Linear(hidden_size_A, hidden_size_A)
        self.map3 = nn.Linear(hidden_size_A, output_size_A)

    def forward(self, x):
        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        #x = torch.sigmoid(self.map3(x)) #has to be sigmoidor anything else between [0, 1]
        x = F.relu(self.map3(x))
        return x.reshape(-1)

    # def update(self, d_score, optimizer):
    #     #loss = criterion(d_score, -(torch.tensor([[5]], dtype=torch.float))) #minus in front of the second term for stochastic ascent
    #     loss = criterion(d_score, (torch.tensor([[5]], dtype=torch.float)))
    #     optimizer.zero_grad()
    #     loss.backward(retain_graph=True)
    #     optimizer.step()
    #     pass


class Critic(nn.Module):
    """
    Aim: Approximates environment dynamics
    takes random policy as input and outputs a scalar reward value
    """
    def __init__(self, input_size_C, hidden_size_C, output_size_C):
        super(Critic, self).__init__()
        self.saved_rewards = []

        self.map1 = nn.Linear(input_size_C, hidden_size_C)
        self.map2 = nn.Linear(hidden_size_C, hidden_size_C)
        self.map3 = nn.Linear(hidden_size_C, output_size_C)

    def forward(self, x):
        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        x = F.relu(self.map3(x))
        #x = self.map3(x)
        return x

    # def update(self, d_score, v_score, optimizer):
    #     v_score = torch.tensor([[v_score]], dtype=torch.float)
    #     loss = criterion(d_score, v_score)
    #     optimizer.zero_grad()
    #     loss.backward(retain_graph=True)
    #     optimizer.step()
    #     pass