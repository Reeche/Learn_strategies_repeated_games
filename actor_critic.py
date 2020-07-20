import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as tdist
import torch.nn.functional as F

torch.manual_seed(0)
criterion = nn.MSELoss()

def get_noise_input_for_actor(size_of_noise):
    """
    :return: Get a random array of size n (int) as input for the Generator/Actor
    """
    n = tdist.Normal(torch.tensor([0.]), torch.tensor([1.]))
    z = torch.t(n.sample((size_of_noise,)))
    return z


class Actor(nn.Module):
    """
    Aim: Generate dominating strategies
    takes a random noise as input (scalar) and outputs a policy
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
        x = F.softmax(self.map3(x))
        return x.reshape(-1)

    def update(self, d_score, optimizer):
        #loss = criterion(d_score, -(torch.tensor([[5]], dtype=torch.float))) #minus in front of the second term for stochastic ascent
        loss = criterion(d_score, (torch.tensor([[5]], dtype=torch.float)))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        pass


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
        x = self.map3(x)
        return x

    def update(self, d_score, v_score, optimizer):
        v_score = torch.tensor([[v_score]], dtype=torch.float)
        loss = criterion(d_score, v_score)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        pass