import os

import mdptoolbox
import numpy as np
import psutil
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import lil_matrix as sparse_lil

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


def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print('{} memory used: {} MB'.format(hint, memory))


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


def convert_matrix_to_sparse(A, H):
    A = np.array([sparse(A[ADOPT]), sparse(A[OVERRIDE]),
                  sparse(A[WAIT]), sparse(A[MATCH])])
    H = np.array([sparse(H[ADOPT]), sparse(H[OVERRIDE]),
                  sparse(H[WAIT]), sparse(H[MATCH])])
    return A, H


def convert_matrix_to_dense(A, H):

    A = np.array([(A[ADOPT]).toarray(), (A[OVERRIDE]).toarray(),
                  (A[WAIT]).toarray(), (A[MATCH]).toarray()])
    H = np.array([(H[ADOPT]).toarray(), (H[OVERRIDE]).toarray(),
                  (H[WAIT]).toarray(), (H[MATCH]).toarray()])
    return A, H


def adjust_reward_with_overpaying(A, H, alpha, rho):
    # reward under action adopt.
    A, H = convert_matrix_to_dense(A, H)
    for a in range(0, rounds):
        for h in range(0, rounds):
            if a == rounds-1 or h == rounds-1:
                index_row_begin = get_index(
                    a, h, IRRELEVANT, rounds, fork_states_num)
                index_row_end = index_row_begin+2
                if a == rounds-1:
                    for index_column_current in [get_index(1, 0, IRRELEVANT, rounds, fork_states_num), get_index(0, 1, IRRELEVANT, rounds, fork_states_num)]:
                        A[ADOPT, index_row_begin:index_row_end + 1,
                            index_column_current] = overpaying_reward_agh(rho, alpha, a, h)
                        H[ADOPT, index_row_begin: index_row_end +
                            1, index_column_current] = 0
                elif h == rounds-1:
                    for index_column_current in [get_index(1, 0, IRRELEVANT, rounds, fork_states_num), get_index(0, 1, IRRELEVANT, rounds, fork_states_num)]:
                        A[ADOPT, index_row_begin: index_row_end + 1,
                            index_column_current] = overpaying_reward_hga(rho, alpha, a, h)
                        H[ADOPT, index_row_begin: index_row_end +
                            1, index_column_current] = 0
    A, H = convert_matrix_to_sparse(A, H)
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

    P = [sparse(P[ADOPT]), sparse(P[OVERRIDE]),
         sparse(P[WAIT]), sparse(P[MATCH])]

    return P


def generate_reward_matrix(states_num, action_num, rounds, fork_states_num, pay_type):
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

            adversary_height, honest_height = 1, 0
            index_column_current = get_index(adversary_height, honest_height,
                                             IRRELEVANT, rounds, fork_states_num)
            H[ADOPT, index_row_begin:index_row_end +
              1, index_column_current] += h

            adversary_height, honest_height = 0, 1
            index_column_current = get_index(adversary_height, honest_height,
                                             IRRELEVANT, rounds, fork_states_num)
            H[ADOPT, index_row_begin:index_row_end +
              1, index_column_current] += h

    # reward under action override.
    for a in range(0, rounds-1):
        for h in range(0, rounds-1):
            if a > h:
                index_row_begin = get_index(
                    a, h, IRRELEVANT, rounds, fork_states_num)
                index_row_end = index_row_begin+2

                clear_value_in_diagonal(
                    A[OVERRIDE], range(index_row_begin, index_row_end+1))
                clear_value_in_diagonal(
                    H[OVERRIDE], range(index_row_begin, index_row_end+1))

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
            clear_value_in_diagonal(
                H[WAIT], [index_row])

            A[WAIT, index_row, get_index(
                a-h, 1, RELEVANT, rounds, fork_states_num)] += h

    # reward under action match.
    for a in range(0, rounds-1):
        for h in range(0, rounds-1):
            if a >= h:
                index_row = get_index(
                    a, h, RELEVANT, rounds, fork_states_num)
                clear_value_in_diagonal(
                    A[MATCH], [index_row])
                clear_value_in_diagonal(
                    H[MATCH], [index_row])

                A[WAIT, index_row, get_index(
                    a-h, 1, RELEVANT, rounds, fork_states_num)] += h
    A, H = convert_matrix_to_sparse(A, H)
    return A, H


if __name__ == "__main__":
    epsilon = 0.0001
    # There are three different fork for the sanme height combination.

    # four actions: adopt, override, wait, match.
    action_num, fork_states_num = 4, 3
    gamma = 0
    # for alpha in range(350, 500, 25):
    # for alpha in range(400, 500, 25):
    # for alpha in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
    # for alpha in [0.425, 0.45, 0.475]:
    for alpha in [0.45, 0.475]:
        if alpha <= 0.4:
            rounds = 80-1
        else:
            rounds = 90
        states_num = rounds*rounds*3
        P = generate_probability_matrix(
            states_num, action_num, rounds, fork_states_num, alpha, gamma)
        low, high = 0, 1
        # UNDERPAYING
        A, H = generate_reward_matrix(
            states_num, action_num, rounds, fork_states_num, UNDERPAYING)
        while high-low > epsilon/8:
            rho = (low+high)/2
            R = []
            show_memory_info(
                "underpaying alpha: {}, rho: {}".format(alpha, rho))
            # generate Reward with different rho.
            for action in [ADOPT, OVERRIDE, WAIT, MATCH]:
                R.append((1-rho)*A[action]-rho*H[action])
            rvi = mdptoolbox.mdp.RelativeValueIteration(P, R)
            rvi.run()

            if rvi.average_reward > 0:
                low = rho
            else:
                high = rho
        print("low bound: alpha: {}, gamma: {}, rho: {}".format(alpha, gamma, rho))

        # OVERPAYING
        high = min(rho+0.1, 1)
        low = rho
        A, H = generate_reward_matrix(
            states_num, action_num, rounds, fork_states_num, OVERPAYING)
        while high-low > epsilon/8:
            rho = (low+high)/2
            A_current, H_current = adjust_reward_with_overpaying(
                A, H, alpha, rho)
            A_current, H_current = convert_matrix_to_sparse(
                A_current, H_current)
            show_memory_info(
                "overpaying alpha: {}, rho: {}".format(alpha, rho))
            R = []
            # generate Reward with different rho.
            for action in [ADOPT, OVERRIDE, WAIT, MATCH]:
                R.append((1-rho)*A_current[action]-rho*H_current[action])
            rvi = mdptoolbox.mdp.RelativeValueIteration(P, R)
            rvi.run()
            if rvi.average_reward > 0:
                low = rho
            else:
                high = rho
        print("upper bound: alpha: {}, gamma: {}, rho: {}".format(alpha, gamma, rho))
