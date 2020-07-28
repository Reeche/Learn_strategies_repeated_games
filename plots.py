import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_actual_predicted_reward(actual_reward, predicted_reward):
    plt.plot(actual_reward, label='Actual reward')
    plt.plot(predicted_reward, label='Predicted reward')
    plt.legend()
    plt.savefig('rewards_13.png')
    plt.show()

def plot_policy(policy):
    for i in range(len(policy)):
        policy[i] = policy[i].detach().numpy()

    numpy_array = np.array(policy)
    transpose = numpy_array.T
    transpose_list = transpose.tolist()

    plt.plot(transpose_list[0], label='P(C|CC)')
    plt.plot(transpose_list[1], label='P(C|CD)')
    plt.plot(transpose_list[2], label='P(C|DC)')
    plt.plot(transpose_list[3], label='P(C|DD)')
    plt.legend()
    plt.savefig('policy_13.png')
    plt.show()



#todo: tidy up all plots

### initialisation
# # 0 means cooperation/Stag/swerve chicken
# # 1 means defect/Hare/straight hawk
#
# p = [11/13, 1/2, 7/26, 0] #epsilon 0.01 leads to convergence
#
#
# ### initialise randomly
# x_action = np.random.randint(2)
# y_action = np.random.randint(2)
#
#
# ### choose the environment
# env = IPD()
# #env = optionalIPD()
# #env = staghunt()
# #env = chicken()
#
# jointstate = env.get_state_id(x_action, y_action) # outputs a scalar
# Q_class = GeneralQ()

def plotqvalues_twoactions(jointstate, Q_class, env, strategy):
    Q_class.reset()
    x_reward = []
    y_reward = []
    qvalues1 = []
    qvalues2 = []
    qvalues3 = []
    qvalues4 = []
    qvalues5 = []
    qvalues6 = []
    qvalues7 = []
    qvalues8 = []

    # training
    for _ in range(0, 4000000): #6000000 IPD
        oldstate = jointstate
        prob = strategy[oldstate]
        #x_action = int(choices([0, 1, 2], prob)[0])
        x_action = int(np.random.choice([0, 1], 1, p=[prob, 1-prob]))
        y_action = Q_class.get_action(oldstate)

        # update new state
        jointstate = env.get_state_id(x_action, y_action)  # new state

        # update reward
        reward_y = env.reward_y(jointstate)
        Q_class.update(oldstate, y_action, reward_y, jointstate)

        qvalues1.append(Q_class.print_q()[:][0][0])
        qvalues2.append(Q_class.print_q()[:][0][1])
        qvalues3.append(Q_class.print_q()[:][1][0])
        qvalues4.append(Q_class.print_q()[:][1][1])
        qvalues5.append(Q_class.print_q()[:][2][0])
        qvalues6.append(Q_class.print_q()[:][2][1])
        qvalues7.append(Q_class.print_q()[:][3][0])
        qvalues8.append(Q_class.print_q()[:][3][1])

    q_plot = pd.DataFrame({'Run 1': qvalues1, 'Run 2': qvalues2, 'Run 3': qvalues3,
                           'Run 4': qvalues4, 'Run 5': qvalues5, 'Run 6': qvalues6,
                           'Run 7': qvalues7})
    sns.set()
    plt.ylabel('Q-values')
    plt.xlabel('Steps')
    plt.plot(q_plot)
    #plt.title('Fixed strategy {}'.format(strategy))
    plt.show()

    # testing
    for _ in range(0, 1000):
        oldstate = jointstate
        prob = strategy[oldstate]
        #x_action = int(choices([0, 1, 2], prob)[0])
        x_action = int(np.random.choice([0, 1], 1, p=[prob, 1-prob])) #1-prob because 0 denotes cooperation
        y_action = Q_class.act(oldstate)

        # update new state
        jointstate = env.get_state_id(x_action, y_action) #new state

        # append reward but no update
        reward_x = env.reward_x(jointstate)
        reward_y = env.reward_y(jointstate)
        x_reward.append(reward_x)
        y_reward.append(reward_y)

    print("reward x: ", np.mean(x_reward), "---- reward y: ", np.mean(y_reward))
    #Q_class.print_q()
    #return np.mean(x_reward), np.mean(y_reward)
    pass


#plotqvalues_twoactions(jointstate, Q_class, env, p)


def plotqvalues_threeactions(jointstate, Q_class, env, strategy):
    Q_class.reset()
    x_reward = []
    y_reward = []
    qvalues1 = []
    qvalues2 = []
    qvalues3 = []
    qvalues4 = []
    qvalues5 = []
    qvalues6 = []
    qvalues7 = []
    qvalues8 = []
    qvalues9 = []
    qvalues10 = []
    qvalues11 = []
    qvalues12 = []
    qvalues13 = []
    qvalues14 = []
    qvalues15 = []
    qvalues16 = []
    qvalues17 = []
    qvalues18 = []
    qvalues19 = []
    qvalues20 = []
    qvalues21 = []
    qvalues22 = []
    qvalues23 = []
    qvalues24 = []
    qvalues25 = []
    qvalues26 = []
    qvalues27 = []

    # training
    for _ in range(0, 6000000): #6000000 IPD
        oldstate = jointstate
        prob = strategy[oldstate]
        x_action = int(choices([0, 1, 2], prob)[0])
        y_action = Q_class.get_action(oldstate)

        # update new state
        jointstate = env.get_state_id(x_action, y_action)  # new state

        # update reward
        reward_y = env.reward_y(jointstate)
        Q_class.update(oldstate, y_action, reward_y, jointstate)

        qvalues1.append(Q_class.print_q()[:][0][0])
        qvalues2.append(Q_class.print_q()[:][0][1])
        qvalues3.append(Q_class.print_q()[:][0][2])
        qvalues4.append(Q_class.print_q()[:][1][0])
        qvalues5.append(Q_class.print_q()[:][1][1])
        qvalues6.append(Q_class.print_q()[:][1][2])
        qvalues7.append(Q_class.print_q()[:][2][0])
        qvalues8.append(Q_class.print_q()[:][2][1])
        qvalues9.append(Q_class.print_q()[:][2][2])
        qvalues10.append(Q_class.print_q()[:][3][0])
        qvalues11.append(Q_class.print_q()[:][3][1])
        qvalues12.append(Q_class.print_q()[:][3][2])
        qvalues13.append(Q_class.print_q()[:][4][0])
        qvalues14.append(Q_class.print_q()[:][4][1])
        qvalues15.append(Q_class.print_q()[:][4][2])
        qvalues16.append(Q_class.print_q()[:][5][0])
        qvalues17.append(Q_class.print_q()[:][5][1])
        qvalues18.append(Q_class.print_q()[:][5][2])
        qvalues19.append(Q_class.print_q()[:][6][0])
        qvalues20.append(Q_class.print_q()[:][6][1])
        qvalues21.append(Q_class.print_q()[:][6][2])
        qvalues22.append(Q_class.print_q()[:][7][0])
        qvalues23.append(Q_class.print_q()[:][7][1])
        qvalues24.append(Q_class.print_q()[:][7][2])
        qvalues25.append(Q_class.print_q()[:][8][0])
        qvalues26.append(Q_class.print_q()[:][8][1])
        qvalues27.append(Q_class.print_q()[:][8][2])

    q_plot = pd.DataFrame({'Qvalue 1': qvalues1, 'Qvalue 2': qvalues2, 'Qvalue 3': qvalues3,
                           'Qvalue 4': qvalues4, 'Qvalue 5': qvalues5, 'Qvalue 6': qvalues6,
                           'Qvalue 7': qvalues7, 'Qvalue 8': qvalues8, 'Qvalue 9': qvalues9,
                           'Qvalue 10': qvalues10, 'Qvalue 11': qvalues11, 'Qvalue 12': qvalues12,
                           'Qvalue 13': qvalues13, 'Qvalue 14': qvalues14, 'Qvalue 15': qvalues15,
                           'Qvalue 16': qvalues16, 'Qvalue 17': qvalues17, 'Qvalue 18': qvalues18})
    sns.set()
    plt.ylabel('Q-values')
    plt.xlabel('Steps')
    plt.plot(q_plot)
    #plt.title('Fixed strategy {}'.format(strategy))
    plt.show()

    # testing
    for _ in range(0, 1000):
        oldstate = jointstate
        prob = strategy[oldstate]
        x_action = int(choices([0, 1, 2], prob)[0])
        y_action = Q_class.act(oldstate)

        # update new state
        jointstate = env.get_state_id(x_action, y_action) #new state

        # append reward but no update
        reward_x = env.reward_x(jointstate)
        reward_y = env.reward_y(jointstate)
        x_reward.append(reward_x)
        y_reward.append(reward_y)

    print("reward x: ", np.mean(x_reward), "---- reward y: ", np.mean(y_reward))
    #Q_class.print_q()
    return np.mean(x_reward), np.mean(y_reward)

#plotqvalues_threeactions(jointstate, Q_class, env, p)
