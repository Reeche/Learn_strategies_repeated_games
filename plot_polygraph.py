import matplotlib.pyplot as plt
import random
import numpy as np
from env import *

# choose the environment
env = IPD()


#p_x = [0.9269, 0.0129, 0.0502, 0.0416]
p_x = [0.80002, 0.59999, 0.10004, 0] #PD dominant


def twofixed(jointstate, env, x_strategy):
    average_x = []
    average_y = []
    i = 0
    for _ in range(3000):
        i = i + 1
        x_reward = []
        y_reward = []

        # for every run generate a new set of p_y
        p_y = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

        for _ in range(0, 50000):
            oldstate = jointstate
            prob_x = x_strategy[oldstate] # takes i-th position of the probability vector
            prob_y = p_y[oldstate]
            x_action = int(np.random.choice([0, 1], 1, p=[prob_x, 1-prob_x]))
            y_action = int(np.random.choice([0, 1], 1, p=[prob_y, 1-prob_y]))

            # update new state
            jointstate = env.get_state_id(x_action, y_action) #new state

            reward_x = env.reward_x(jointstate)
            reward_y = env.reward_y(jointstate)
            x_reward.append(reward_x)
            y_reward.append(reward_y)

        print(i, "-th iteration. ---- reward x: ", np.mean(x_reward), "---- reward y: ", np.mean(y_reward))
        average_x.append(np.mean(x_reward))
        average_y.append(np.mean(y_reward))

    return average_x, average_y


x_action = np.random.randint(2)
y_action = np.random.randint(2)
jointstate = env.get_state_id(x_action, y_action)

x_reward, y_reward = twofixed(jointstate, env, p_x)
plt.axes()

points = [[0, 5], [1, 1], [5, 0], [3, 3]]
line = plt.Polygon(points, closed=True, fill=False, edgecolor='r')

plt.scatter(x_reward, y_reward)
plt.gca().add_patch(line)

plt.axis('scaled')
plt.show()