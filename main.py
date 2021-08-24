import datetime
import os
from abc import abstractclassmethod

import mdptoolbox
import numpy as np
import psutil
from numpy.lib.function_base import delete
from scipy import sparse
from scipy.sparse import coo_matrix as sparse_coo
from scipy.sparse import csr_matrix as sparse_csr
from scipy.sparse.sputils import matrix

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

actions_lib = [ADOPT, OVERRIDE, WAIT, MATCH]


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


def convert_data_to_sparse_matrix(P, states_num):
    P_result = []
    for action in actions_lib:
        P_result.append(dict())
        for row_num in range(states_num):
            P_result[action][row_num] = dict()

    for action in actions_lib:
        for index in range(len(P[action]["rows"])):
            row = P[action]["rows"][index]
            col = P[action]["cols"][index]
            value = P[action]["values"][index]
            if col in P_result[action][row].keys():
                P_result[action][row][col] += value
            else:
                P_result[action][row][col] = value

    for action in actions_lib:
        current_rows = []
        current_cols = []
        current_values = []
        for index in range(len(P[action]["rows"])):
            current_rows.append(P[action]["rows"][index])
            current_cols.append(P[action]["cols"][index])
            current_values.append(P[action]["values"][index])
        P[action] = sparse_coo((current_values, (current_rows, current_cols)), shape=[
            states_num, states_num]).tocsr()
    return P


# def adjust_reward_with_overpaying(A, H, alpha, rho):
#     # reward under action adopt.
#     A, H = convert_matrix_to_dense(A, H)

#     for a in range(0, rounds):
#         for h in range(0, rounds):
#             if a == rounds-1 or h == rounds-1:
#                 index_row_begin = get_index(
#                     a, h, IRRELEVANT, rounds, fork_states_num)
#                 index_row_end = index_row_begin+2
#                 if a == rounds-1:
#                     for index_column_current in [get_index(1, 0, IRRELEVANT, rounds, fork_states_num), get_index(0, 1, IRRELEVANT, rounds, fork_states_num)]:
#                         A[ADOPT, index_row_begin:index_row_end + 1,
#                             index_column_current] = overpaying_reward_agh(rho, alpha, a, h)
#                         H[ADOPT, index_row_begin: index_row_end +
#                             1, index_column_current] = 0
#                 elif h == rounds-1:
#                     for index_column_current in [get_index(1, 0, IRRELEVANT, rounds, fork_states_num), get_index(0, 1, IRRELEVANT, rounds, fork_states_num)]:
#                         A[ADOPT, index_row_begin: index_row_end + 1,
#                             index_column_current] = overpaying_reward_hga(rho, alpha, a, h)
#                         H[ADOPT, index_row_begin: index_row_end +
#                             1, index_column_current] = 0
#     A, H = convert_matrix_to_sparse(A, H)
#     return A, H


def generate_probability_matrix(states_num,  rounds, fork_states_num, alpha, gamma):

    # init the data.
    P = []
    for action in actions_lib:
        P.append({"rows": np.zeros([states_num*5]),
                  "cols": np.zeros([states_num*5]),
                  "values": np.zeros([states_num*5])}
                 )

    ordered_list = [i for i in range(states_num)]
    for action in [OVERRIDE, WAIT, MATCH]:
        P[action]["rows"][0:states_num] = ordered_list
        P[action]["cols"][0:states_num] = ordered_list
        P[action]["values"][0:states_num] = 1
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
    P[ADOPT]["rows"][0:states_num] = ordered_list
    P[ADOPT]["cols"][0:states_num] = [index_column_current] * states_num
    P[ADOPT]["values"][0:states_num] = [alpha] * states_num

    # with probablity 1 - alpha to (0, 1, IRRELEVANT) I am not sure if it should be (0, 1, RELEVANT)
    # 可能是错的。
    adversary_height, honest_height = 0, 1
    index_column_current = get_index(adversary_height, honest_height,
                                     IRRELEVANT, rounds, fork_states_num)
    P[ADOPT]["rows"][states_num:2*states_num] = ordered_list
    P[ADOPT]["cols"][states_num:2*states_num] = [index_column_current] * states_num
    P[ADOPT]["values"][states_num:2*states_num] = [alpha] * states_num
    # probability under action override.
    # only works for a > h.
    # I am wrong at the first attempt. PLZ remember it !!!
    cursor = {OVERRIDE: states_num, WAIT: states_num, MATCH: states_num}

    for a in range(rounds-1):
        for h in range(rounds-1):
            # get row.
            index_row_begin = get_index(
                a, h, IRRELEVANT, rounds, fork_states_num)
            index_row_end = index_row_begin+fork_states_num
            values_zero = [0]*fork_states_num
            index_row_interval = [i for i in range(
                index_row_begin, index_row_end)]
            if a > h:
                # clear diagonal.
                P[OVERRIDE]["values"][index_row_begin:index_row_end] = values_zero

                # with probablity alpha to ((a - h), 0, 0).
                index_column_current = get_index(
                    (a-h), 0, IRRELEVANT, rounds, fork_states_num)
                P[OVERRIDE]["rows"][cursor[OVERRIDE]:cursor[OVERRIDE] +
                                    fork_states_num] = index_row_interval
                P[OVERRIDE]["cols"][cursor[OVERRIDE]:cursor[OVERRIDE] +
                                    fork_states_num] = [index_column_current]*fork_states_num
                P[OVERRIDE]["values"][cursor[OVERRIDE]:cursor[OVERRIDE] +
                                      fork_states_num] = [alpha]*fork_states_num

                cursor[OVERRIDE] += fork_states_num
                # with probablity alpha to ((a - h s- 1), 1, 1).

                index_column_current = get_index(
                    (a-h-1), 1, RELEVANT, rounds, fork_states_num)
                P[OVERRIDE]["rows"][cursor[OVERRIDE]:cursor[OVERRIDE] +
                                    fork_states_num] = index_row_interval
                P[OVERRIDE]["cols"][cursor[OVERRIDE]:cursor[OVERRIDE] +
                                    fork_states_num] = [index_column_current]*fork_states_num
                P[OVERRIDE]["values"][cursor[OVERRIDE]:cursor[OVERRIDE] +
                                      fork_states_num] = [1-alpha]*fork_states_num
                cursor[OVERRIDE] += fork_states_num

            # probability under action wait.
            # clear diagonal.
            P[WAIT]["values"][index_row_begin:index_row_end] = values_zero

            # WAIT:IRRELEVANT
            P[WAIT]["rows"][cursor[WAIT]:cursor[WAIT] + 2] = [index_row_begin]*2
            P[WAIT]["cols"][cursor[WAIT]:cursor[WAIT] + 2] = [
                get_index(a+1, h, IRRELEVANT, rounds, fork_states_num),
                get_index(a, h+1, RELEVANT, rounds, fork_states_num)
            ]
            P[WAIT]["values"][cursor[WAIT]: cursor[WAIT] + 2] = [alpha, 1-alpha]

            cursor[WAIT] += 2

            # WAIT:RELEVANT
            P[WAIT]["rows"][cursor[WAIT]:cursor[WAIT] + 2] = [index_row_begin+1]*2
            P[WAIT]["cols"][cursor[WAIT]:cursor[WAIT] + 2] = [
                get_index(a+1, h, IRRELEVANT, rounds, fork_states_num),
                get_index(a, h+1, RELEVANT, rounds, fork_states_num)
            ]
            P[WAIT]["values"][cursor[WAIT]: cursor[WAIT] + 2] = [alpha, 1-alpha]

            cursor[WAIT] += 2

            # WAIT:ACTIVE
            if a >= h:
                P[WAIT]["rows"][cursor[WAIT]:cursor[WAIT] + 3] = [index_row_begin+2]*3
                P[WAIT]["cols"][cursor[WAIT]:cursor[WAIT] + 3] = [
                    get_index(a+1, h, ACTIVE, rounds, fork_states_num),
                    get_index(a-h, 1, RELEVANT, rounds, fork_states_num),
                    get_index(a, h+1, RELEVANT, rounds, fork_states_num)
                ]
                P[WAIT]["values"][cursor[WAIT]: cursor[WAIT] +
                                  3] = [alpha, gamma*(1-alpha), (1-gamma)*(1-alpha)]
                cursor[WAIT] += 3

            # probability under action match.
            if a >= h:
                # clear diagonal.
                P[MATCH]["values"][index_row_begin+1] = 0

                # a, h, relevant.
                P[MATCH]["rows"][cursor[MATCH]
                    :cursor[MATCH] + 3] = [index_row_begin+1]*3
                P[MATCH]["cols"][cursor[MATCH]:cursor[MATCH] + 3] = [
                    get_index(a+1, h, ACTIVE, rounds, fork_states_num),
                    get_index(a-h, 1, RELEVANT, rounds, fork_states_num),
                    get_index(a, h+1, RELEVANT, rounds, fork_states_num)
                ]
                P[MATCH]["values"][cursor[MATCH]: cursor[MATCH] +
                                   3] = [alpha, gamma*(1-alpha), (1-gamma)*(1-alpha)]

                cursor[MATCH] += 3
    show_memory_info("state P init.")
    for action in [OVERRIDE, WAIT, MATCH]:
        for key in P[action].keys():
            P[action][key] = np.delete(
                P[action][key], [i for i in range(cursor[action], states_num*5)])
    # get the result.
    P = convert_data_to_sparse_matrix(P, states_num)
    return P


def generate_reward_matrix(states_num, rounds, fork_states_num, pay_type):

    A, H = [], []
    for action in actions_lib:
        A.append({"rows": np.zeros([states_num*5]),
                  "cols": np.zeros([states_num*5]),
                  "values": np.zeros([states_num*5])}
                 )
        H.append({"rows": np.zeros([states_num*5]),
                  "cols": np.zeros([states_num*5]),
                  "values": np.zeros([states_num*5])}
                 )

    ordered_list = [i for i in range(states_num)]
    for action in [OVERRIDE, WAIT, MATCH]:
        H[action]["rows"][0:states_num] = ordered_list
        H[action]["cols"][0:states_num] = ordered_list
        H[action]["values"][0:states_num] = 100000

    cursor_A = {ADOPT: 0, OVERRIDE: 0, WAIT: 0, MATCH: 0}
    cursor_H = {ADOPT: 0, OVERRIDE: states_num,
                WAIT: states_num, MATCH: states_num}

    values_zero = [0]*fork_states_num
    # reward under action adopt.
    for a in range(0, rounds):
        for h in range(0, rounds):
            index_row_begin = get_index(
                a, h, IRRELEVANT, rounds, fork_states_num)
            index_row_end = index_row_begin+fork_states_num

            adversary_height, honest_height = 1, 0
            index_column_current = get_index(adversary_height, honest_height,
                                             IRRELEVANT, rounds, fork_states_num)

            H[ADOPT]["rows"][cursor_H[ADOPT]:cursor_H[ADOPT]+fork_states_num] = [i for i in range(
                index_row_begin, index_row_end)]
            H[ADOPT]["cols"][cursor_H[ADOPT]:cursor_H[ADOPT]+fork_states_num] = [
                index_column_current] * fork_states_num
            H[ADOPT]["values"][cursor_H[ADOPT]:cursor_H[ADOPT] +
                               fork_states_num] = [h] * fork_states_num
            cursor_H[ADOPT] += 3

            adversary_height, honest_height = 0, 1
            index_column_current = get_index(adversary_height, honest_height,
                                             IRRELEVANT, rounds, fork_states_num)

            H[ADOPT]["rows"][cursor_H[ADOPT]:cursor_H[ADOPT]+fork_states_num] = [i for i in range(
                index_row_begin, index_row_end)]
            H[ADOPT]["cols"][cursor_H[ADOPT]:cursor_H[ADOPT]+fork_states_num] = [
                index_column_current] * fork_states_num
            H[ADOPT]["values"][cursor_H[ADOPT]:cursor_H[ADOPT] +
                               fork_states_num] = [h] * fork_states_num
            cursor_H[ADOPT] += 3

    for a in range(0, rounds-1):
        for h in range(0, rounds-1):
            if a > h:
                index_row_begin = get_index(
                    a, h, IRRELEVANT, rounds, fork_states_num)
                index_row_end = index_row_begin+fork_states_num

                H[OVERRIDE]["values"][index_row_begin:index_row_end] = values_zero
                # reward under action override.

                index_column_current = get_index(
                    (a-h), 0, IRRELEVANT, rounds, fork_states_num)

                A[OVERRIDE]["rows"][cursor_A[OVERRIDE]:cursor_A[OVERRIDE] +
                                    fork_states_num] = [i for i in range(index_row_begin, index_row_end)]
                A[OVERRIDE]["cols"][cursor_A[OVERRIDE]:cursor_A[OVERRIDE] +
                                    fork_states_num] = [index_column_current]*fork_states_num
                A[OVERRIDE]["values"][cursor_A[OVERRIDE]:cursor_A[OVERRIDE] +
                                      fork_states_num] = [h+1]*fork_states_num
                cursor_A[OVERRIDE] += 3

                index_column_current = get_index(
                    (a-h-1), 1, RELEVANT, rounds, fork_states_num)

                A[OVERRIDE]["rows"][cursor_A[OVERRIDE]:cursor_A[OVERRIDE] +
                                    fork_states_num] = [i for i in range(index_row_begin, index_row_end)]
                A[OVERRIDE]["cols"][cursor_A[OVERRIDE]:cursor_A[OVERRIDE] +
                                    fork_states_num] = [index_column_current]*fork_states_num
                A[OVERRIDE]["values"][cursor_A[OVERRIDE]:cursor_A[OVERRIDE] +
                                      fork_states_num] = [h+1]*fork_states_num
                cursor_A[OVERRIDE] += 3
            # reward under action wait.
            if a >= h:
                # ACTIVE
                index_row = get_index(
                    a, h, ACTIVE, rounds, fork_states_num)

                H[WAIT]["values"][index_row] = 0

                A[WAIT]["rows"][cursor_A[WAIT]] = index_row
                A[WAIT]["cols"][cursor_A[WAIT]] = get_index(
                    a-h, 1, RELEVANT, rounds, fork_states_num)
                A[WAIT]["values"][cursor_A[WAIT]] = h
                cursor_A[WAIT] += 1

            # reward under action match.
            if a >= h:
                index_row = get_index(
                    a, h, RELEVANT, rounds, fork_states_num)
                H[MATCH]["values"][index_row] = 0

                A[MATCH]["rows"][cursor_A[MATCH]] = index_row
                A[MATCH]["cols"][cursor_A[MATCH]] = get_index(
                    a-h, 1, RELEVANT, rounds, fork_states_num)
                A[MATCH]["values"][cursor_A[MATCH]] = h
                cursor_A[MATCH] += 1
    A = convert_data_to_sparse_matrix(A, states_num)
    H = convert_data_to_sparse_matrix(H, states_num)
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
    # for alpha in [0.45, 0.475]:
    for alpha in [0.45]:
        if alpha <= 0.4:
            rounds = 80-1
        else:
            rounds = 20
        # rounds = 20
        states_num = rounds*rounds*3
        P = generate_probability_matrix(
            states_num, rounds, fork_states_num, alpha, gamma)
        low, high = 0, 1
        # UNDERPAYING
        A, H = generate_reward_matrix(
            states_num, rounds, fork_states_num, UNDERPAYING)
        while high-low > epsilon/8:
            rho = (low+high)/2
            R = [None] * 4
            show_memory_info(
                "underpaying alpha: {}, rho: {}".format(alpha, rho))
            # generate Reward with different rho.
            for action in [ADOPT, OVERRIDE, WAIT, MATCH]:
                R[action] = (1-rho)*A[action]-rho*H[action]
            rvi = mdptoolbox.mdp.RelativeValueIteration(P, R)
            rvi.run()

            if rvi.average_reward > 0:
                low = rho
            else:
                high = rho
        print("low bound: alpha: {}, gamma: {}, rho: {}".format(alpha, gamma, rho))

        # # OVERPAYING
        # high = min(rho+0.1, 1)
        # low = rho
        # A, H = generate_reward_matrix(
        #     states_num, action_num, rounds, fork_states_num, OVERPAYING)
        # while high-low > epsilon/8:
        #     rho = (low+high)/2
        #     A_current, H_current = adjust_reward_with_overpaying(
        #         A, H, alpha, rho)
        #     A_current, H_current = convert_matrix_to_sparse(
        #         A_current, H_current)
        #     show_memory_info(
        #         "overpaying alpha: {}, rho: {}".format(alpha, rho))
        #     R = []
        #     # generate Reward with different rho.
        #     for action in [ADOPT, OVERRIDE, WAIT, MATCH]:
        #         R.append((1-rho)*A_current[action]-rho*H_current[action])
        #     rvi = mdptoolbox.mdp.RelativeValueIteration(P, R)
        #     rvi.run()
        #     if rvi.average_reward > 0:
        #         low = rho
        #     else:
        #         high = rho
        # print("upper bound: alpha: {}, gamma: {}, rho: {}".format(alpha, gamma, rho))
