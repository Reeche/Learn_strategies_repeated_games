import numpy as np
from helper import *
from environment import *
from random import choices

# 1 is cooperate, 0 is defect
p_x = [0.80002, 0.59999, 0.10004, 0] #the dominant strategy according to press, dyson with phi = 0.0001, X = 10000
p_y = fixedstrategies()

### initialise randomly
x_action = np.random.randint(2)
y_action = np.random.randint(2)

# choose the environment
env = ipd()
#env = optionalIPD()
#env = staghunt()
#env = chicken()

jointstate = env.get_state_id(x_action, y_action) # outputs a scalar


def twofixed(jointstate, env, x_strategy, y_strategy):
    x_reward = []
    y_reward = []
    for _ in range(0, 500000): # 500000
        oldstate = jointstate
        prob_x = x_strategy[oldstate] # takes i-th position of the probability vector
        prob_y = y_strategy[oldstate]
        x_action = int(np.random.choice([0, 1], 1, p=[prob_x, 1-prob_x]))
        y_action = int(np.random.choice([0, 1], 1, p=[prob_y, 1-prob_y]))

        # update new state
        jointstate = env.get_state_id(x_action, y_action) #new state

        reward_x = env.reward_x(jointstate)
        reward_y = env.reward_y(jointstate)
        x_reward.append(reward_x)
        y_reward.append(reward_y)

    print("reward x: ", np.mean(x_reward), "---- reward y: ", np.mean(y_reward), "; y strategy is ", y_strategy)
    return np.mean(x_reward), np.mean(y_reward), y_strategy

for i in range(0, len(p_y)):
    reward_x, reward_y, strategy_y = twofixed(jointstate, env, p_x, p_y[i])

def twofixed_threeactions(jointstate, env, x_strategy, y_strategy):
    x_reward = []
    y_reward = []
    for _ in range(0, 100000): # 100000
        oldstate = jointstate
        prob_x = x_strategy[oldstate] # takes i-th position of the probability vector
        prob_y = y_strategy[oldstate]
        x_action = int(choices([0, 1, 2], prob_x)[0])
        y_action = int(choices([0, 1, 2], prob_y)[0])

        # update new state
        jointstate = env.get_state_id(x_action, y_action) #new state

        reward_x = env.reward_x(jointstate)
        reward_y = env.reward_y(jointstate)
        x_reward.append(reward_x)
        y_reward.append(reward_y)

    print("reward x: ", np.mean(x_reward), "---- reward y: ", np.mean(y_reward), "; y strategy is ", y_strategy)
    return np.mean(x_reward), np.mean(y_reward), y_strategy

#twofixed_threeactions(jointstate, env, p_x, p_y)