import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from collections import namedtuple
import torch.nn.functional as F

from env import IPD
from tableQ import TableQAgent
from actor_critic import Actor, Critic
from interaction import interaction
from plots import plot_actual_predicted_reward, plot_policy

from logger import root_logger

logger = logging.getLogger(__name__)

def main():
    i = 0
    predicted_reward_list = []
    actual_reward_list = []
    generated_policy_list = []
    for _ in range(number_of_episodes):
        x_action = np.random.randint(2)
        y_action = np.random.randint(2)
        init_state = env.get_state_id(x_action, y_action)

        for _ in range(length_of_episode):
            Q_class = TableQAgent()

            # Input the state into the Actor to generate a policy
            generated_policy = Actor_class.forward(torch.FloatTensor([init_state]))

            # Input policy of Actor into Critic to predict a score
            predicted_reward = Critic_class.forward(generated_policy)

            # Actor save the policy and corresponding predicted value
            Actor_class.saved_actions.append(SavedAction(generated_policy, predicted_reward))

            # Training
            # R = 0
            saved_actions = Actor_class.saved_actions
            policy_losses = []  # list to save actor (policy) loss
            value_losses = []  # list to save critic (value) loss
            returns = []  # list to save the true values

            # calculate the true value using rewards returned from the environment
            true_reward_x, _ = interaction(init_state, Q_class, env, generated_policy)
            print("GENERATED POLICY: ", generated_policy.detach().numpy().ravel())
            print("PREDICTED REWARD: ", predicted_reward.item())
            print("TRUE REWARD: ", true_reward_x)

            # Critic observes the actual reward and saves them
            Critic_class.saved_rewards.append([true_reward_x])

            for r in Critic_class.saved_rewards[::-1]:
                # for i in range(len(Critic_class.saved_rewards)):
                # calculate the discounted value
                # R = r + (gamma * R * 0)
                # print("R", R)
                returns.insert(0, torch.tensor(r, dtype=torch.float))

            returns = torch.tensor(returns)
            # if len(returns) > 1:  # to avoid std of 1 value
            #     returns = (returns - returns.mean()) / (returns.std())
            # print("RETURNS", returns)

            for (log_prob, predicted_value), R in zip(saved_actions,
                                                      returns):  # (generated policy, predicated_reward), true reward
                advantage = R - predicted_value.item()  # true reward - predicated reward
                diff = 5 - R

                # calculate actor (policy) loss
                policy_losses.append(-log_prob * (
                        advantage + diff))  # difference between actual and predicted + difference until maximum reward

                # calculate critic (value) loss using L1 smooth loss
                print("----- Predicted Reward: ", predicted_value.item(), " VS Actual reward: ", R.item(), "-----")
                value_losses.append(F.smooth_l1_loss(predicted_value, torch.tensor([R])))

        # save those values for plotting
        predicted_reward_list.insert(len(predicted_reward_list), predicted_reward)
        actual_reward_list.insert(len(actual_reward_list), true_reward_x)
        generated_policy_list.insert(len(generated_policy_list), generated_policy)

        # reset gradients
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss_actor = torch.stack(policy_losses).sum()
        loss_critic = torch.stack(value_losses).sum()
        # print("----- The generated policy is: ", generated_policy.detach().numpy().ravel(), "-----")
        # print(loss_actor)
        # print(loss_critic)

        # perform backprop
        loss_actor.backward(retain_graph=True)
        loss_critic.backward(retain_graph=True)
        optimizer_actor.step()
        optimizer_critic.step()

        # reset rewards and action buffer
        del Critic_class.saved_rewards[:]
        del Actor_class.saved_actions[:]

        logger.info("The {}-episode with generated policy {}, predicated reward of {} and actual reward of {}"
                    .format(i, generated_policy, predicted_reward, true_reward_x))

        print("######################### END OF EPISODE ", i, "####################################")
        i += 1
    return predicted_reward_list, actual_reward_list, generated_policy_list


if __name__ == '__main__':
    # choose the environment and the agent
    env = IPD()
    # Q_class = TableQAgent()

    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
    gamma = 0.99
    eps = np.finfo(np.float32).eps.item()

    input_size_A, hidden_size_A, output_size_A = 1, 2, 4
    input_size_C, hidden_size_C, output_size_C = 4, 2, 1

    Actor_class = Actor(input_size_A, hidden_size_A, output_size_A)
    Critic_class = Critic(input_size_C, hidden_size_C, output_size_C)

    optimizer_actor = optim.Adam(Actor_class.parameters(), lr=3e-2, betas=(0.9, 0.999))
    optimizer_critic = optim.Adam(Critic_class.parameters(), lr=3e-2)

    number_of_episodes = 100
    # can be thought of as batch size, i.e. how often I want to update the network with the same (state) settings
    length_of_episode = 4
    predicted_reward_list, actual_reward_list, generated_policy_list = main()
    plot_actual_predicted_reward(predicted_reward_list, actual_reward_list)
    plot_policy(generated_policy_list)


