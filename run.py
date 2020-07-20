import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from random import choices

from actor_critic import get_noise_input_for_actor, Actor, Critic
from interaction import interaction
from tableQ import TableQAgent
from env import IPD

# initialise
qlearner = TableQAgent()
env = IPD()

# get initial state
init_state = 1

# get noise input for actor
noise = get_noise_input_for_actor(10)

# input the noise into the Actor to get a policy
generated_policy = Actor.forward((noise))

# input policy of Actor into Critic
predicted_reward = Critic.forward(generated_policy)

# Get the RL score of the generated policy
actual_reward_x, actual_reward_y = interaction(init_state, qlearner, env, generated_policy)

# Train the Critic
def train_critic(noise):
    policy = Actor_obj.forward(noise)
    true_score_x, true_reward_y = rl_score(policy)
    for _ in range(0, 100):
        # policy is input for C-network and outputs score D
        predicted_score = Critic_obj.forward(policy)

        # update C-network using MSE loss (D - V)
        Critic_obj.update(predicted_score, true_score_x, critic_optimizer)

    return predicted_score



# update actor
def train_actor(noise):
    for _ in range(1500):
        # Input the random number into A-network and outputs a policy
        policy = Actor_obj.forward(noise)
        # get predicted score through C network
        reward = Critic_obj.forward(policy)

        # update A-network using MSE loss (D - V)
        Actor_obj.update(reward, actor_optimizer)
    return reward, policy


# policy is input for the RL agent and outputs a V score
def rl_score(policy_tensor):
    """
    Takes the policy as input and outputs a score (X's and Y's average reward)
    :param policy:
    :return: X and Y average reward
    """
    policy = policy_tensor.detach().numpy()
    print("fixed policy that is playing against the RL agent: ", policy)
    Q_class = TableQAgent()
    x_action = np.random.randint(2)
    y_action = np.random.randint(2)
    jointstate = env.get_state_id(x_action, y_action)
    averagy_X, average_Y = interaction(jointstate, Q_class, env, policy)
    return averagy_X, average_Y


def main():
    reward_ls = []
    i = 0
    n = 10 #size of random input into the Actor
    for _ in range(30):

        # train a critic model
        for _ in range(16):
            predicted_score = train_critic(get_input(n))
            print("predicted_score in the critic network", predicted_score)

        # train the actor
        reward, policy = train_actor(get_input(n))

        print("reward after optimising through actor", reward)
        reward_ls.append(reward)
        i = i + 1
        print("------------------", i, "-th iteration ----------------------")
    print("reward list", reward_ls)
    print("policy", policy)
    plt.plot(reward_ls)
    plt.show()


if __name__ == "__main__":
    main()


##########################################################################################

def select_action(current_state, SavedAction):
    state = torch.from_numpy(current_state).float()
    probs, state_value = Actor(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    Actor.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take
    return action.item()



def finish_episode(eps, optimizer_actor, optimizer_critic):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = Actor.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in Critic.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer_actor.zero_grad()
    optimizer_critic.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss_actor = torch.stack(policy_losses).sum()
    loss_critic = torch.stack(value_losses).sum()

    # perform backprop
    loss_actor.backward()
    loss_critic.backward()
    optimizer_actor.step()
    optimizer_critic.step()

    # reset rewards and action buffer
    del Critic.rewards[:]
    del Actor.saved_actions[:]


def main():
    running_reward = 10
    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
    env = IPD()
    eps = np.finfo(np.float32).eps.item()
    optimizer_actor = optim.Adam(Actor.parameters(), lr=3e-2)
    optimizer_critic = optim.Adam(Critic.parameters(), lr=3e-2)

    # run number of episodes
    for i_episode in count(1):

        # reset environment and episode reward
        current_state = env.reset()
        ep_reward = 0

        # length of each episode
        for t in range(1, 10000):

            # todo: what is needed is state and reward of agent X, interaction.py gets the reward

            # select action for agent Y based on policy
            y_action = select_action(current_state, SavedAction)

            # selection action for agent X from policy
            prob = strategy[current_state]
            x_action = int(choices([0, 1], [prob, 1 - prob])[0])

            # take the action
            state = env.get_state_id(y_action, x_action)
            reward_y = env.reward_y(y_action, x_action)
            #reward_x = env.reward_x(y_action, x_action)

            Critic.rewards.append(reward_y)
            ep_reward += reward_y

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode(eps, optimizer_actor, optimizer_critic)




if __name__ == '__main__':
    main()