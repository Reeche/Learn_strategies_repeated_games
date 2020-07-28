import numpy as np


def getstrategy(qtable):
    p = []
    for i in range(len(qtable)):
        m = max(qtable[i])
        p.append([i for i, j in enumerate(qtable[i]) if j == m][0])
    return p


def fixedstrategies():
    """
    The 16 fixed strategies
    :return:
    """
    p_1 = [0, 0, 0, 0]
    p_2 = [0, 0, 0, 1]
    p_3 = [0, 0, 1, 0]
    p_4 = [0, 1, 0, 0]
    p_5 = [1, 0, 0, 0]
    p_6 = [1, 1, 0, 0]
    p_7 = [1, 0, 1, 0]
    p_8 = [1, 0, 0, 1]
    p_9 = [0, 1, 1, 0]
    p_10 = [0, 1, 0, 1]
    p_11 = [0, 0, 1, 1]
    p_12 = [1, 1, 1, 0]
    p_13 = [1, 1, 0, 1]
    p_14 = [1, 0, 1, 1]
    p_15 = [0, 1, 1, 1]
    p_16 = [1, 1, 1, 1]
    p = [p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9, p_10,
         p_11, p_12, p_13, p_14, p_15, p_16]
    return p


def transitionmatrix(p, q):
    """
    :param p: has to be 4x1
    :param q: has to be 4x1
    :return: 4x4 matrix
    """
    matrix = np.zeros((4, 4))
    matrix[0, 0] = p[0] * q[0]
    matrix[0, 1] = p[0] * (1 - q[0])
    matrix[0, 2] = (1 - p[0]) * q[0]
    matrix[0, 3] = (1 - p[0]) * (1 - q[0])
    matrix[1, 0] = p[1] * q[2]
    matrix[1, 1] = p[1] * (1 - q[2])
    matrix[1, 2] = (1 - p[1]) * q[2]
    matrix[1, 3] = (1 - p[1]) * (1 - q[2])
    matrix[2, 0] = p[2] * q[1]
    matrix[2, 1] = p[2] * (1 - q[1])
    matrix[2, 2] = (1 - p[2]) * q[1]
    matrix[2, 3] = (1 - p[2]) * (1 - q[1])
    matrix[3, 0] = p[3] * q[3]
    matrix[3, 1] = p[3] * (1 - q[3])
    matrix[3, 2] = (1 - p[3]) * q[3]
    matrix[3, 3] = (1 - p[3]) * (1 - q[3])
    return matrix


def derivatives(R, S, T, P, v, q):
    p_1 = R * v[0] * q[0] + S * v[0] * (1 - q[0]) - T * v[0] * q[0] - P * v[0] + P * v[0] * q[0]
    p_2 = R * v[1] * q[2] + S * v[1] * (1 - q[2]) - T * v[1] * q[2] - P * v[1] + P * v[1] * q[2]
    p_3 = R * v[2] * q[1] + S * v[2] * (1 - q[1]) - T * v[2] * q[1] - P * v[2] + P * v[2] * q[1]
    p_4 = R * v[3] * q[3] + S * v[3] * (1 - q[3]) - T * v[3] * q[3] - P * v[3] + P * v[3] * q[3]
    return p_1, p_2, p_3, p_4

def derivatives_transposed(R, S, T, P, v, q):
    d1 = (v[0] * q[0] + v[1] * (1 - q[0]) - v[2] * q[0] - v[3] + v[3] * q[0]) * R
    d2 = (v[0] * q[2] + v[1] * (1 - q[2]) - v[2] * q[2] - v[3] + v[3] * q[2]) * S
    d3 = (v[0] * q[1] + v[1] * (1 - q[1]) - v[2] * q[1] - v[3] + v[3] * q[1]) * T
    d4 = (v[0] * q[3] + v[1] * (1 - q[3]) - v[2] * q[3] - v[3] + v[3] * q[3]) * P
    return d1, d2, d3, d4