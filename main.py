import datetime

import mdptoolbox
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import size
from numpy.core.numeric import Inf
from numpy.lib.index_tricks import IndexExpression
from scipy.sparse import csr_matrix as sparse

IRRELEVANT, RELEVANT, ACTIVE = 0, 1, 2

ADOPT, OVERRIDE, WAIT, MATCH = 0, 1, 2, 3

UNDERPAYING = 0
OVERPAYING = 1

states = dict()
states[0] = "IRRELEVANT"
states[1] = "RELEVANT"
states[2] = "ACTIVE"


actions = dict()
actions[0] = "ADOPT"
actions[1] = "OVERRIDE"
actions[2] = "WAIT"
actions[3] = "MATCH"


def overpaying_reward_agh(rho, alpha, a, h):
    return (1-rho)*(alpha*(1-alpha)/pow(1-2*alpha, 2))+1/2*((a-h)/(1-2*alpha)+a+h)


def overpaying_reward_hga(rho, alpha, a, h):
    return (1-pow(alpha/(1-alpha), h-a))*(-rho*h)+pow(alpha/(1-alpha), h-a)*(1-rho)*(alpha*(1-alpha)/pow(1-2*alpha, 2)+(h-a)/(1-2*alpha))


def get_index(a, h, f, rounds, fork_states_num):
    return a * rounds * fork_states_num + h * fork_states_num + f


def get_state(index, rounds, fork_states_num):
    a, remainder = divmod(index, rounds*fork_states_num)
    h, f = divmod(remainder, fork_states_num)
    return "({}, {}, {})".format(a, h, states[f])


def clear_value_in_diagonal(matrix, indexs):
    for index in indexs:
        matrix[index][index] = 0


def monitor_value(matrix, a, h, f, rounds, fork_states_num):
    index = get_index(a, h, f, rounds, fork_states_num)
    return matrix[index][index]


def adopt_probability(matrix, rounds, fork_states_num, alpha):
    for a in range(0, rounds):
        for h in range(0, rounds):
            if a == rounds-1 or h == rounds-1:
                # clear to zero.
                index_row_begin = get_index(
                    a, h, IRRELEVANT, rounds, fork_states_num)
                index_row_end = index_row_begin+2
                clear_value_in_diagonal(
                    matrix, range(index_row_begin, index_row_end+1))

                adversary_height, honest_height = 1, 0
                index_column_current = get_index(adversary_height, honest_height,
                                                 IRRELEVANT, rounds, fork_states_num)
                matrix[index_row_begin: index_row_end +
                       1:, index_column_current] += alpha

                adversary_height, honest_height = 0, 1
                index_column_current = get_index(adversary_height, honest_height,
                                                 IRRELEVANT, rounds, fork_states_num)
                matrix[index_row_begin: index_row_end +
                       1:, index_column_current] += 1-alpha
    return matrix


def adopt_reward(A, H, rounds, fork_states_num, alpha, rho, pay_type):
    for a in range(0, rounds):
        for h in range(0, rounds):
            if a == rounds-1 or h == rounds-1:
                index_row_begin = get_index(
                    a, h, IRRELEVANT, rounds, fork_states_num)
                index_row_end = index_row_begin+2

                clear_value_in_diagonal(
                    H, range(index_row_begin, index_row_end+1))

                adversary_height, honest_height = 1, 0
                index_column_current = get_index(adversary_height, honest_height,
                                                 IRRELEVANT, rounds, fork_states_num)
                if pay_type == UNDERPAYING:
                    H[index_row_begin: index_row_end +
                      1:, index_column_current] = -100000
                else:
                    if a == rounds-1:
                        A[index_row_begin: index_row_end +
                          1:, index_column_current] += overpaying_reward_agh(rho, alpha, a, h)
                    else:
                        A[index_row_begin: index_row_end +
                          1:, index_column_current] += overpaying_reward_hga(rho, alpha, a, h)

                adversary_height, honest_height = 0, 1
                index_column_current = get_index(adversary_height, honest_height,
                                                 IRRELEVANT, rounds, fork_states_num)
                if pay_type == UNDERPAYING:
                    H[index_row_begin: index_row_end +
                      1:, index_column_current] = -100000
                else:
                    if a == rounds-1:
                        A[index_row_begin: index_row_end +
                          1:, index_column_current] += overpaying_reward_agh(rho, alpha, a, h)
                    else:
                        A[index_row_begin: index_row_end +
                          1:, index_column_current] += overpaying_reward_hga(rho, alpha, a, h)

    return A, H


def generate_probability_matrix(states_num, action_num, rounds, fork_states_num, alpha, gamma):

    P = np.zeros([action_num, states_num, states_num])

    for action in [OVERRIDE, WAIT, MATCH]:
        np.fill_diagonal(P[action], 1)
    # the structure of probability is (A, H, F)
    # irrelevant = 0, relevant =1, active = 2
    # (0, 0, irrelevant)
    # (0, 0, relevant)
    # (0, 0, active)
    # (0, 1, irrelevant)
    # (0, 1, relevant)
    # (0, 1, active)
    # ...
    # (0, 75, active)
    # (1, 0, irrelevant)

    # probability under action adopt.
    # with probablity alpha to (1, 0, IRRELEVANT) position
    adversary_height, honest_height = 1, 0
    index_column_current = get_index(adversary_height, honest_height,
                                     IRRELEVANT, rounds, fork_states_num)
    P[ADOPT, :, index_column_current] += alpha

    # with probablity 1 - alpha to (0, 1, IRRELEVANT) I am not sure if it should be (0, 1, RELEVANT)
    # 可能是错的。
    adversary_height, honest_height = 0, 1
    index_column_current = get_index(adversary_height, honest_height,
                                     IRRELEVANT, rounds, fork_states_num)
    P[ADOPT, :, index_column_current] += 1-alpha

    # probability under action override.
    # only works for a > h.
    # I am wrong at the first attempt. PLZ remember it !!!
    for a in range(0, rounds-1):
        for h in range(0, rounds-1):
            if a > h:
                index_row_begin = get_index(
                    a, h, IRRELEVANT, rounds, fork_states_num)
                index_row_end = get_index(
                    a, h, ACTIVE, rounds, fork_states_num)

                clear_value_in_diagonal(
                    P[OVERRIDE], range(index_row_begin, index_row_end+1))

                # with probablity alpha to ((a - h), 0, 0).
                index_column_current = get_index(
                    (a-h), 0, IRRELEVANT, rounds, fork_states_num)
                P[OVERRIDE, index_row_begin:index_row_end+1,
                    index_column_current] += alpha

                # with probablity alpha to ((a - h - 1), 1, 1).
                index_column_current = get_index(
                    (a-h-1), 1, RELEVANT, rounds, fork_states_num)
                P[OVERRIDE, index_row_begin:index_row_end+1,
                    index_column_current] += 1-alpha

    # probability under action wait.
    for a in range(0, rounds-1):
        for h in range(0, rounds-1):
            # IRRELEVANT
            index_row = get_index(
                a, h, IRRELEVANT, rounds, fork_states_num)
            clear_value_in_diagonal(P[WAIT], [index_row])
            P[WAIT, index_row, get_index(
                a+1, h, IRRELEVANT, rounds, fork_states_num)] += alpha
            P[WAIT, index_row, get_index(
                a, h+1, RELEVANT, rounds, fork_states_num)] += 1-alpha

            # RELEVANT
            index_row += 1
            clear_value_in_diagonal(P[WAIT], [index_row])
            P[WAIT, index_row, get_index(
                a+1, h, IRRELEVANT, rounds, fork_states_num)] += alpha
            P[WAIT, index_row, get_index(
                a, h+1, RELEVANT, rounds, fork_states_num)] += 1-alpha

            # ACTIVE
            index_row += 1
            clear_value_in_diagonal(P[WAIT], [index_row])
            P[WAIT, index_row, get_index(
                a+1, h, ACTIVE, rounds, fork_states_num)] += alpha
            #  这里错了，要注意。
            P[WAIT, index_row, get_index(
                a-h, 1, RELEVANT, rounds, fork_states_num)] += gamma*(1-alpha)
            P[WAIT, index_row, get_index(
                a, h+1, RELEVANT, rounds, fork_states_num)] += (1-gamma)*(1-alpha)

    # probability under action match.
    for a in range(0, rounds-1):
        for h in range(0, rounds-1):
            if a >= h:
                index_row = get_index(
                    a, h, RELEVANT, rounds, fork_states_num)
                clear_value_in_diagonal(P[MATCH], [index_row])
                P[MATCH, index_row, get_index(
                    a+1, h, ACTIVE, rounds, fork_states_num)] += alpha
                P[MATCH, index_row, get_index(
                    a-h, 1, RELEVANT, rounds, fork_states_num)] += gamma*(1-alpha)
                P[MATCH, index_row, get_index(
                    a, h+1, RELEVANT, rounds, fork_states_num)] += (1-gamma)*(1-alpha)

    for action in [OVERRIDE, WAIT, MATCH]:
        P[action] = adopt_probability(
            P[action], rounds, fork_states_num, alpha)

    P = [sparse(P[ADOPT]), sparse(P[OVERRIDE]),
         sparse(P[WAIT]), sparse(P[MATCH])]

    return P


def generate_reward_matrix(states_num, action_num, rounds, fork_states_num, alpha, rho, pay_type):
    A = np.zeros([action_num, states_num, states_num])
    H = np.zeros([action_num, states_num, states_num])

    for action in [OVERRIDE, WAIT, MATCH]:
        np.fill_diagonal(H[action], 100000)

    # reward under action adopt.
    for a in range(0, rounds):
        for h in range(0, rounds):
            index_row_begin = get_index(
                a, h, IRRELEVANT, rounds, fork_states_num)
            index_row_end = index_row_begin+2

            clear_value_in_diagonal(
                A[ADOPT], range(index_row_begin, index_row_end+1))

            adversary_height, honest_height = 1, 0
            index_column_current = get_index(adversary_height, honest_height,
                                             IRRELEVANT, rounds, fork_states_num)
            if pay_type == UNDERPAYING:
                H[ADOPT, index_row_begin:index_row_end +
                    1, index_column_current] += h
            else:
                if a == rounds-1:
                    A[ADOPT, index_row_begin:index_row_end +
                      1, index_column_current] += overpaying_reward_agh(rho, alpha, a, h)
                else:
                    A[ADOPT, index_row_begin:index_row_end +
                      1, index_column_current] += overpaying_reward_hga(rho, alpha, a, h)

            adversary_height, honest_height = 0, 1
            index_column_current = get_index(adversary_height, honest_height,
                                             IRRELEVANT, rounds, fork_states_num)
            if pay_type == UNDERPAYING:
                H[ADOPT, index_row_begin:index_row_end +
                    1, index_column_current] += h
            else:
                if a == rounds-1:
                    A[ADOPT, index_row_begin:index_row_end +
                      1, index_column_current] += overpaying_reward_agh(rho, alpha, a, h)
                else:
                    A[ADOPT, index_row_begin:index_row_end +
                      1, index_column_current] += overpaying_reward_hga(rho, alpha, a, h)

    # reward under action override.
    for a in range(0, rounds-1):
        for h in range(0, rounds-1):
            if a > h:
                index_row_begin = get_index(
                    a, h, IRRELEVANT, rounds, fork_states_num)
                index_row_end = index_row_begin+2

                clear_value_in_diagonal(
                    A[OVERRIDE], range(index_row_begin, index_row_end+1))

                index_column_current = get_index(
                    (a-h), 0, IRRELEVANT, rounds, fork_states_num)
                A[OVERRIDE, index_row_begin:index_row_end+1,
                    index_column_current] += h+1
                index_column_current = get_index(
                    (a-h-1), 1, RELEVANT, rounds, fork_states_num)
                A[OVERRIDE, index_row_begin:index_row_end+1,
                    index_column_current] += h+1

    # reward under action wait.
    for a in range(0, rounds-1):
        for h in range(0, rounds-1):
            # ACTIVE
            index_row = get_index(
                a, h, ACTIVE, rounds, fork_states_num)
            clear_value_in_diagonal(
                A[WAIT], [index_row])
            A[WAIT, index_row, get_index(
                a-h, 1, RELEVANT, rounds, fork_states_num)] += h

    # reward under action match.
    for a in range(0, rounds-1):
        for h in range(0, rounds-1):
            if a >= h:
                index_row = get_index(
                    a, h, RELEVANT, rounds, fork_states_num)
                A[MATCH, index_row, get_index(
                    a-h, 1, RELEVANT, rounds, fork_states_num)] += h

    for action in [OVERRIDE, WAIT, MATCH]:
        A[action], H[action] = adopt_reward(A[action], H[action],
                                            rounds, fork_states_num, alpha, rho, pay_type)

    # print((1-rho)*A[ADOPT])
    # print(H[ADOPT])
    # print(rho)
    # print((1-rho)*A[ADOPT].multiply(-rho*H[ADOPT]))
    return A, H


if __name__ == "__main__":
    epsilon = pow(10, -4)
    rounds = 4

    # There are three different fork for the sanme height combination.
    states_num = rounds*rounds*3
    # four actions: adopt, override, wait, match.
    action_num = 4
    gamma = 0
    fork_states_num = 3
    # for alpha in range(350, 500, 25):
    for alpha in [400]:
        alpha /= 1000
        # generate P.
        P = generate_probability_matrix(
            states_num, action_num, rounds, fork_states_num, alpha, gamma)
        low, high = 0, 1
        # UNDERPAYING
        while high-low > epsilon/8:
            rho = (low+high)/2
            # print("current_rho: {}".format(rho))
            R = []
            # generate Reward with different rho.
            A, H = generate_reward_matrix(
                states_num, action_num, rounds, fork_states_num, alpha, rho, UNDERPAYING)
            for action in [ADOPT, OVERRIDE, WAIT, MATCH]:
                A_prime = sparse((1-rho)*A[action])
                H_prime = sparse(-rho*H[action])
                R.append(A_prime._add_sparse(H_prime))

            rvi = mdptoolbox.mdp.RelativeValueIteration(P, R)
            for action in [ADOPT, OVERRIDE, WAIT, MATCH]:
                column_names = [get_state(index, rounds, fork_states_num)
                                for index in range(0, states_num)]
                row_name = [get_state(index, rounds, fork_states_num)
                            for index in range(0, states_num)]
                df = pd.DataFrame(R[action].todense(), columns=column_names, index=row_name)
                df.to_csv('R-{}.csv'.format(actions[action]), sep='\t')
            break
            rvi.run()
            if rvi.average_reward > 0:
                low = rho
            else:
                high = rho
        print("alpha: {}, gamma: {}, rho: {}".format(alpha, gamma, rho))

    # # OVERPAYING
    # high = min(rho+0.1, 1)

    # while high-low > epsilon/8:
    #     rho = (low+high)/2
    #     print("current_rho {}".format(rho))

    #     # generate Reward with different rho.
    #     R = generate_reward_matrix(
    #         states_num, action_num, rounds, fork_states_num, alpha, rho, OVERPAYING)

    #     rvi = mdptoolbox.mdp.RelativeValueIteration(P, R)
    #     rvi.run()
    #     if rvi.average_reward > 0:
    #         low = rho
    #     else:
    #         high = rho
    # print("upper bound: ", rho)
