import numpy as np
from random import choices

from actor_critic import Actor, Critic
from tableQ import TableQAgent
import sys

# todo: add random seed
def interaction(init_state, Q_class, env, strategy):
    """
    This function models the interaction between the Actor (policy) and the RL

    :param init_state:
    :param Q_class: the Q_learning agent
    :param env: the environment. Is either IPD, Stag-Hunt or Chicken
    :param strategy: the fixed_strategy of the fixed agent X
    :return: Score of the RL agent
    """
    x_reward = []
    y_reward = []
    for _ in range(0, 5):
        Q_class.reset()


        # print("The fixed strategy is: ", strategy)

        # training
        next_state = init_state
        for _ in range(0, 5000):
            current_state = next_state
            prob = strategy[current_state]

            # choose action for both agents
            x_action = int(choices([0, 1], [prob, 1 - prob])[0]) #todo: prob.sample()
            y_action = Q_class.get_action(current_state)

            # get the new state depending on the actions
            next_state = env.get_state_id(x_action, y_action)  # new state

            # update reward
            reward_y = env.reward_y(next_state)
            Q_class.update(current_state, y_action, reward_y, next_state)

        # testing
        next_state = init_state
        temp_x_reward = []
        temp_y_reward = []
        for _ in range(0, 100):
            current_state = next_state
            prob = strategy[current_state]
            x_action = int(choices([0, 1], [prob, 1 - prob])[0])
            y_action = Q_class.get_action(current_state)

            # update new state
            next_state = env.get_state_id(x_action, y_action)  # new state

            # append reward but no update
            reward_x = env.reward_x(next_state)
            reward_y = env.reward_y(next_state)
            temp_x_reward.append(reward_x)
            temp_y_reward.append(reward_y)
        x_reward.append(np.mean(temp_x_reward))
        y_reward.append(np.mean(temp_y_reward))
    #print("The average reward for fixed agent x: ", np.mean(x_reward), "---- The average reward for learning agent y: ", np.mean(y_reward))
    # Q_class.print_q()
    return np.mean(x_reward), np.mean(y_reward)

