import numpy as np
from hyperparameters import seed

np.random.seed(seed)

class TableQAgent():
    # learning rate should be 0.01, epsilon should be 0.15 WITHOUT epsilon_decay
    def __init__(self, num_state=4, num_action=2, gamma=0.99, epsilon=0.15, lr=0.01, epsilon_decay=1):
        self.num_state = num_state
        self.num_action = num_action
        self.epsilon = epsilon
        self.inital_epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.reset()

    def reset(self):
        # self.q = [[(np.random.random() - 0.5) * 100 for j in range(self.num_action)] for i in range(self.num_state)]
        # self.q = [[0 for j in range(self.num_action)] for i in range(self.num_state)]
        self.q = np.zeros((self.num_state, self.num_action))
        self.epsilon = self.inital_epsilon

    # get action
    def get_action(self, state, training):
        if training:
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.num_action)
                # print('Random action ' + str(action))
                return action
            else:
                action = np.argmax(self.q[state])
                return action
        else:
            action = np.argmax(self.q[state])
            return action


    def update(self, state, action, reward, next_state, done=False):
        max_q_action = 0
        for next_action in range(self.num_action):
            if self.q[next_state][next_action] > self.q[next_state][max_q_action]:
                max_q_action = next_action
        if done:
            diff_q = reward - self.q[state][action]
        else:
            diff_q = reward + self.gamma * self.q[next_state][max_q_action] - self.q[state][action]
        self.q[state][action] = self.q[state][action] + self.lr * diff_q
        self.epsilon *= self.epsilon_decay

    def print_q(self):
        print(self.q)
        return self.q

    def get_policy(self):
        policy = []
        for state in range(self.num_state):
            max_action = 0
            for action in range(self.num_action):
                if self.q[state][action] > self.q[state][max_action]:
                    max_action = action
            policy.append(max_action)
        return policy

    def set_q(self, q):
        self.q = q