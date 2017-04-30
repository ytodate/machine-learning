# coding: utf-8
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pylab as plt
import random
import sys
import math

# impossible actions < 0
# possible actions == 0
# reword > 0
# r state/action
r = np.array([
    [-1, -1, 0, -1, -1] #sb
    , [-10, -1, 0, -1, -1] #sl
    , [-1, 0, -1, 0, -1] #se
    , [-1, -1, 0, -1, 10] #sf
    , [-1, -1,  0, -1, -1] #st
], dtype='f')

# q value
q = np.zeros_like(r)

LEARNING_NUM = 10000
ALPHA = 0.1
GAMMA = 0.9
GOAL_STATE = [0, 4] #index
EPOCH_NUM = 100

# seeds
random.seed(a=6006)
np.random.seed(seed=6006)

# actions
EPSILON = 0.3
T = 5

action_methods = [
    "RANDOM"
    , "GREEDY"
    , "E-GREEDY"
    , "BOLTZMANN"
]


class QLearning(object):
    def __init__(self):
        return

# 学習して、qを更新する
    def learn(self, action_method="RANDOM"):
        q_values = []
        epoch = 0
        state = self._getInitialState()
        print("epoch %s th q is \n %s" % (epoch, np.round(q, 4)))

        for i in range(LEARNING_NUM):
            possible_actions = self._getPossibleActionsFromState(state)

            action = self.action_select(state, possible_actions, method=action_method)

            next_state = action

            next_possible_actions = self._getPossibleActionsFromState(next_state)

            max_q_next_state_action = self._getMaxQvalueFromStateAndPossibleActions(next_state, next_possible_actions)

            q[state, action] = q[state, action] + ALPHA * (r[state, action] + GAMMA * max_q_next_state_action - q[state, action])

            # update state
            state = next_state

            # if agent reached a goal state, restart from random state
            if state in GOAL_STATE:
                state = self._getInitialState()
                epoch += 1
                q_values.append(q[3, 4])

                if epoch % (EPOCH_NUM/10) == 0:
                    print("epoch %s th q is \n %s" % (epoch, np.round(q, 4)))

            if epoch >= EPOCH_NUM:
                return q_values


    def _getInitialState(self, init=int(r.shape[0]/2)):
        return init

    def _getRandomState(self):
        return random.randint(0, r.shape[0] - 1)

    def _getPossibleActionsFromState(self, state):
        # validate state
        if state < 0 or state >= r.shape[0]: sys.exit('invaid state: %d' % state)
        # 移動可能なstateをリストで返す
        return list(np.where(np.array(r[state] != -1)))[0]

    def _getMaxQvalueFromStateAndPossibleActions(self, state, possible_actions):
        # 移動可能なstateの中で、qが最大となるところを返す
        return max([q[state][i] for i in (possible_actions)])

    def dumpQvalue(self):
        print("q = %s" % np.rint(q))  # convert float to int for redability


    # actions
    def action_select(self, state, possible_actions, method="RANDOM"):
        methods = {
            "RANDOM": self.random(possible_actions)
            , "GREEDY": self.greedy(state, possible_actions)
            , "E-GREEDY": self.e_greedy(state, possible_actions)
            , "BOLTZMANN": self.boltzmann(state, possible_actions)
        }
        action = methods[method]
        return action

    def random(self, possible_actions):
        act = random.choice(possible_actions)
        return act

    def greedy(self, state, possible_actions):
        max_q = 0
        actions = []
        for action in possible_actions:
            if q[state][action] > max_q:
                actions = [action]
                max_q = q[state][action]
            elif q[state][action] == max_q:
                actions.append(action)

        # get a best action from candidates randomly
        act = random.choice(actions)
        return act

    def e_greedy(self, state, possible_actions):
        rand = random.random()

        if rand < EPSILON:
            return self.greedy(state, possible_actions)

        return self.random(possible_actions)

    def boltzmann(self, state, possible_actions):

        bolt_list = [math.exp(q[state, a] / T) for a in possible_actions]
        p_list = [bolt / sum(bolt_list) for bolt in bolt_list]
        action = np.random.choice(possible_actions, 1, p=p_list)[0]
        return action

    def roulette(self, state, possible_actions):
        roul_list = [q[state, a] for a in possible_actions]
        p_list = [roul / sum(roul_list) for roul in roul_list]
        action = np.random.choice(possible_actions, 1, p=p_list)[0]
        return action

    # test
    def test(self, start_state=0, action_method="GREEDY"):
        print "===== START ====="
        state = start_state
        while state not in GOAL_STATE:
            print "current state: %d" % state
            possible_actions = self._getPossibleActionsFromState(state)
            action = self.action_select(state, possible_actions, method=action_method)
            print "-> choose action: %d" % action
            state = action  # in this example, action value is same as next state
        print "state is %d, GOAL!!" % state



if __name__ == "__main__":
    # learning
    QL = QLearning()
    QL.learn()

    q_values_list = []
    for i, method in enumerate(action_methods):
        q = np.zeros_like(r)
        q_values = QL.learn(action_method=method)
        q_values_list.append(q_values)

        # test
        QL.test(start_state=2)

    plt.figure()
    for i, q_values in enumerate(q_values_list):
        plt.plot(q_values_list[i], marker="o", label=action_methods[i])
    plt.xlabel("epoch")
    plt.ylabel("Q(s_f,E)")
    plt.legend(loc="lower right")
    plt.savefig("plot.png")
    plt.close()

    QL.dumpQvalue()


