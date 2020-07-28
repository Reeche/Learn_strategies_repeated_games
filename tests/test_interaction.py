import unittest
import numpy as np

from env import IPD
from tableQ import TableQAgent
from interaction import interaction

class TestInteraction(unittest.TestCase):
    def test_interaction(self):
        x_action = np.random.randint(2)
        y_action = np.random.randint(2)
        env = IPD()
        init_state = env.get_state_id(x_action, y_action)
        Q_class = TableQAgent()

        # epislon 0.15 without decay works well with 200k steps!

        strategy = [0.9999, 0.1111, 0.9999, 0.1111] # is approx. 3 - 3
        # strategy = [1., 0., 1., 0.]
        # strategy = [0.9999, 0.9999, 0.9999, 0.9999]  # is approx 0 - 5-+
        # strategy = [0.1111, 0.1111, 0.1111, 0.1111] #is approx 1 - 1

        # strategy = [11/13, 1/2, 7/26, 0] #3.65139 -- The average reward for learning agent y:  1.8763999999999998
        for _ in range(5):
            reward_x, reward_y = interaction(init_state, Q_class, env, strategy)
            # print("The average reward for fixed agent x: ", reward_x, "-- The average reward for learning agent y: ",
            #       reward_y)
        self.assertTrue(len(strategy) == 4)



