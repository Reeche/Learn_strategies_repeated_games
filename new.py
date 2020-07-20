import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
import torch.nn.functional as F

from env import IPD
from tableQ import TableQAgent
from actor_critic import Actor, Critic
from interaction import interaction


# todo: set random seeds

def main():
    for _ in range(number_of_episodes):
        for _ in range(length_of_episode):
            # Input the state into the Actor to generate a policy
            generated_policy = Actor_class.forward(torch.FloatTensor([init_state]))

            # Input policy of Actor into Critic to predict a score
            predicted_reward = Critic_class.forward(generated_policy)
            Critic_class.saved_rewards.append(predicted_reward)

            # Actor save the policy and corresponding reward
            # Actor_class.saved_actions.append(generated_policy, predicted_reward)
            Actor_class.saved_actions.append(SavedAction(generated_policy, predicted_reward))

            # Training
            saved_actions = Actor_class.saved_actions
            policy_losses = []  # list to save actor (policy) loss
            value_losses = []  # list to save critic (value) loss
            returns = []  # list to save the true values

            # calculate the true value using rewards returned from the environment
            true_reward_x, true_reward_y = interaction(init_state, Q_class, env, generated_policy)
            for r in Critic_class.saved_rewards[::-1]:
                # calculate the discounted value
                true_reward_x = r + gamma * true_reward_x
                returns.insert(0, true_reward_x)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            for (log_prob, value), true_reward_x in zip(saved_actions, returns):
                advantage = true_reward_x - value.item()

                # calculate actor (policy) loss
                policy_losses.append(-log_prob * advantage)

                # calculate critic (value) loss using L1 smooth loss
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([true_reward_x])))

        # reset gradients
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss_actor = torch.stack(policy_losses).sum()
        loss_critic = torch.stack(value_losses).sum()
        print("The generated policy is: ", generated_policy)
        print(loss_actor)
        print(loss_critic)

        # perform backprop
        loss_actor.backward(retain_graph=True)
        loss_critic.backward(retain_graph=True)
        optimizer_actor.step()
        optimizer_critic.step()

        # reset rewards and action buffer
        del Critic_class.saved_rewards[:]
        del Actor_class.saved_actions[:]


if __name__ == '__main__':
    # Initialisation, randomly choose 2 actions
    x_action = np.random.randint(2)
    y_action = np.random.randint(2)

    # choose the environment and the agent
    env = IPD()
    Q_class = TableQAgent()

    init_state = env.get_state_id(x_action, y_action)
    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
    gamma = 0.99
    eps = np.finfo(np.float32).eps.item()

    input_size_A, hidden_size_A, output_size_A = 1, 2, 4
    input_size_C, hidden_size_C, output_size_C = 4, 2, 1

    Actor_class = Actor(input_size_A, hidden_size_A, output_size_A)
    Critic_class = Critic(input_size_C, hidden_size_C, output_size_C)

    optimizer_actor = optim.Adam(Actor_class.parameters(), lr=3e-2)
    optimizer_critic = optim.Adam(Critic_class.parameters(), lr=3e-2)

    number_of_episodes = 100
    length_of_episode = 32 #can be thought of as batch size
    main()
