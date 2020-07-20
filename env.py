
class IPD():
    #T, R, P, S = [5, 3, 1, 0]
    # T, R, P, S = [5, 4, 2, 1]
    #T, R, P, S = [10, 1, -5, -10]
    def __init__(self):
        # 0 = (0, 0) = (C, C);
        # 1 = (0, 1) = (C, D);
        # 2 = (1, 0) = (D, C);
        # 3 = (1, 1) = (D, D)

        self.action_space = 2
        self.state_space = 4

    def reward_x(self, state):
        if state == 2:
            return 5  # T
        if state == 0:
            return 3  # R
        if state == 3:
            return 1  # P
        if state == 1:
            return 0  # S

    def reward_y(self, state):
        if state == 2:
            return 0  # S
        if state == 0:
            return 3  # R
        if state == 3:
            return 1  # P
        if state == 1:
            return 5  # T

    def get_state_id(self, x_state, y_state):
        if x_state == 0 and y_state == 0:
            return 0  # cc
        if x_state == 0 and y_state == 1:
            return 1  # cd
        if x_state == 1 and y_state == 0:
            return 2  # dc
        if x_state == 1 and y_state == 1:
            return 3  # dd

    def decode_state_id(self, state):
        if state == 0:
            return [0, 0] #cc
        if state == 1:
            return [0, 1] #cd
        if state == 2:
            return [1, 0] #dc
        if state == 3:
            return [1, 1] #dd


class StagHunt():
    def __init__(self):
        # 0 = (0, 0) = (S, S);
        # 1 = (0, 1) = (S, H);
        # 2 = (1, 0) = (H, S);
        # 3 = (1, 1) = (H, H)

        self.action_space = 2
        self.state_space = 4

    def reward_x(self, state):
        if state == 2:
            return (2)  # T
        if state == 0:
            return (3)  # R
        if state == 3:
            return (1)  # P
        if state == 1:
            return (0)  # S

    def reward_y(self, state):
        if state == 2:
            return (0)  # T
        if state == 0:
            return (3)  # R
        if state == 3:
            return (1)  # P
        if state == 1:
            return (2)  # S

    def get_state_id(self, x_state, y_state):
        if x_state == 0 and y_state == 0:
            return (0)  # SS
        if x_state == 0 and y_state == 1:
            return (1)  # SH
        if x_state == 1 and y_state == 0:
            return (2)  # HS
        if x_state == 1 and y_state == 1:
            return (3)  # HH

    def decode_state_id(self, state):
        if state == 0:
            return ([0, 0])  # SS
        if state == 1:
            return ([0, 1])  # SH
        if state == 2:
            return ([1, 0])  # HS
        if state == 3:
            return ([1, 1])  # HH

class Chicken():
    def __init__(self):
        # 0 = (0, 0) = (C, C);
        # 1 = (0, 1) = (C, H);
        # 2 = (1, 0) = (H, C);
        # 3 = (1, 1) = (H, H)

        self.action_space = 2
        self.state_space = 4

    def reward_x(self, state):
        if state == 2:
            return 1  # T
        if state == 0:
            return 0  # R
        if state == 3:
            return -1000  # P
        if state == 1:
            return -1  # S

    def reward_y(self, state):
        if state == 2:
            return -1  # T
        if state == 0:
            return 0  # R
        if state == 3:
            return -1000  # P
        if state == 1:
            return 1  # S

    def get_state_id(self, x_state, y_state):
        if x_state == 0 and y_state == 0:
            return 0  # CC
        if x_state == 0 and y_state == 1:
            return 1  # CH
        if x_state == 1 and y_state == 0:
            return 2  # HC
        if x_state == 1 and y_state == 1:
            return 3  # HH

    def decode_state_id(self, state):
        if state == 0:
            return [0, 0] #CC
        if state == 1:
            return [0, 1] #CH
        if state == 2:
            return [1, 0] #HC
        if state == 3:
            return [1, 1] #HH

class optionalIPD():
    # 0 = (0, 0) = (C, C);
    # 1 = (0, 1) = (C, D);
    # 2 = (0, 2) = (C, A);

    # 3 = (1, 0) = (D, C);
    # 2 = (1, 1) = (D, D);
    # 2 = (1, 2) = (D, A);

    # 2 = (2, 0) = (A, C);
    # 2 = (2, 1) = (A, D);
    # 2 = (2, 2) = (A, A);

    def __init__(self):
        self.action_space = 3
        self.state_space = 9
        self.T, self.R, self.L, self.P, self.S = [5, 3, 2, 1, 0]

    def reward_x(self, state):
        if state == 0:
            return self.R  # R
        if state == 1:
            return self.S  # S
        if state == 2:
            return self.L  # L
        if state == 3:
            return self.T  # T
        if state == 4:
            return self.P  # P
        if state == 5:
            return self.L  # L
        if state == 6:
            return self.L  # L
        if state == 7:
            return self.L  # L
        if state == 8:
            return self.L  # L


    def reward_y(self, state):
        if state == 0:
            return self.R  # R
        if state == 1:
            return self.T  # T
        if state == 2:
            return self.L  # L
        if state == 3:
            return self.S  # S
        if state == 4:
            return self.P  # P
        if state == 5:
            return self.L  # L
        if state == 6:
            return self.L  # L
        if state == 7:
            return self.L  # L
        if state == 8:
            return self.L  # L


    def get_state_id(self, x_state, y_state):
        if x_state == 0 and y_state == 0:
            return 0  # cc
        if x_state == 0 and y_state == 1:
            return 1  # cd
        if x_state == 0 and y_state == 2:
            return 2  # dc
        if x_state == 1 and y_state == 0:
            return 3  # dd
        if x_state == 1 and y_state == 1:
            return 4  # cc
        if x_state == 1 and y_state == 2:
            return 5  # cd
        if x_state == 2 and y_state == 0:
            return 6  # dc
        if x_state == 2 and y_state == 1:
            return 7  # dd
        if x_state == 2 and y_state == 2:
            return 8  # dd


    def decode_state_id(self, state):
        if state == 0:
            return [0, 0] #cc
        if state == 1:
            return [0, 1] #cd
        if state == 2:
            return [0, 2] #dc
        if state == 3:
            return [1, 0] #dd
        if state == 4:
            return [1, 1] #cc
        if state == 5:
            return [1, 1] #cd
        if state == 6:
            return [2, 0] #dc
        if state == 7:
            return [2, 1] #dd
        if state == 8:
            return [2, 2] #dd