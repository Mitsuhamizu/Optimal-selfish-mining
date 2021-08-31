import collections
import os
from math import e
from typing import Reversible

import mdptoolbox
import numpy as np
import psutil
from numpy.lib.function_base import delete
from numpy.lib.polynomial import _raise_power
from scipy.sparse import coo_matrix as sparse_coo
from scipy.sparse import csr_matrix as sparse_csr

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


def overpaying_reward_agh(rho, alpha, a, h):
    return (1 - rho) * (alpha * (1 - alpha) / pow(1 - 2 * alpha, 2)) + 1 / 2 * (
        (a - h) / (1 - 2 * alpha) + a + h
    )


def overpaying_reward_hga(rho, alpha, a, h):
    return (1 - pow(alpha / (1 - alpha), h - a)) * (-rho * h) + pow(
        alpha / (1 - alpha), h - a
    ) * (1 - rho) * (
        alpha * (1 - alpha) / pow(1 - 2 * alpha, 2) + (h - a) / (1 - 2 * alpha)
    )


def get_index(a, h, f, rounds, fork_states_num):
    return a * rounds * fork_states_num + h * fork_states_num + f


def get_state(index, rounds, fork_states_num):
    a, remainder = divmod(index, rounds * fork_states_num)
    h, f = divmod(remainder, fork_states_num)
    return "({}, {}, {})".format(a, h, states[f])


def convert_data_to_sparse_matrix(P, states_num):
    for action in actions_lib:
        P[action] = sparse_coo(
            (P[action]["values"], (P[action]["rows"], P[action]["cols"])),
            shape=[states_num, states_num],
        ).tocsr()
    return P


def modify_elements(matrix, index_begin, index_end, rows, cols, values):
    matrix["rows"][index_begin:index_end] = rows
    matrix["cols"][index_begin:index_end] = cols
    matrix["values"][index_begin:index_end] = values
    return matrix


def adjust_reward_with_overpaying(A, H, alpha, rho):
    for a in range(0, rounds):
        for h in range(0, rounds):
            if a == rounds - 1 or h == rounds - 1:
                index_row_begin = get_index(a, h, IRRELEVANT, rounds, fork_states_num)
                index_row_end = index_row_begin + 3
                if a == rounds - 1:
                    for index_column_current in [
                        get_index(1, 0, IRRELEVANT, rounds, fork_states_num),
                        get_index(0, 1, IRRELEVANT, rounds, fork_states_num),
                    ]:
                        A[ADOPT][
                            index_row_begin:index_row_end,
                            index_column_current,
                        ] = overpaying_reward_agh(rho, alpha, a, h)
                        H[ADOPT][
                            index_row_begin:index_row_end,
                            index_column_current,
                        ] = 0
                elif h == rounds - 1:
                    for index_column_current in [
                        get_index(1, 0, IRRELEVANT, rounds, fork_states_num),
                        get_index(0, 1, IRRELEVANT, rounds, fork_states_num),
                    ]:
                        A[ADOPT][
                            index_row_begin:index_row_end,
                            index_column_current,
                        ] = overpaying_reward_hga(rho, alpha, a, h)
                        H[ADOPT][
                            index_row_begin:index_row_end,
                            index_column_current,
                        ] = 0
    return A, H


def generate_matrixs(states_num, rounds, fork_states_num, alpha, gamma):

    # init the data.
    A, H, P = [], [], []
    for action in actions_lib:
        P.append(
            {
                "rows": np.zeros([states_num * 5]),
                "cols": np.zeros([states_num * 5]),
                "values": np.zeros([states_num * 5]),
            }
        )
        A.append(
            {
                "rows": np.zeros([states_num * 5]),
                "cols": np.zeros([states_num * 5]),
                "values": np.zeros([states_num * 5]),
            }
        )
        H.append(
            {
                "rows": np.zeros([states_num * 5]),
                "cols": np.zeros([states_num * 5]),
                "values": np.zeros([states_num * 5]),
            }
        )

    ordered_list = [i for i in range(states_num)]
    for action in [OVERRIDE, WAIT, MATCH]:
        P[action] = modify_elements(
            P[action], 0, states_num, ordered_list, ordered_list, 1
        )
        H[action] = modify_elements(
            H[action], 0, states_num, ordered_list, ordered_list, 1000000
        )

    cursor_A = {ADOPT: 0, OVERRIDE: 0, WAIT: 0, MATCH: 0}
    cursor_H = {ADOPT: 0, OVERRIDE: states_num, WAIT: states_num, MATCH: states_num}
    cursor_P = {OVERRIDE: states_num, WAIT: states_num, MATCH: states_num}
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
    index_column_current = get_index(
        adversary_height, honest_height, IRRELEVANT, rounds, fork_states_num
    )
    P[ADOPT] = modify_elements(
        P[ADOPT],
        0,
        states_num,
        ordered_list,
        [index_column_current] * states_num,
        [alpha] * states_num,
    )

    # with probablity 1 - alpha to (0, 1, IRRELEVANT) I am not sure if it should be (0, 1, RELEVANT)
    # 可能是错的。
    adversary_height, honest_height = 0, 1
    index_column_current = get_index(
        adversary_height, honest_height, IRRELEVANT, rounds, fork_states_num
    )
    P[ADOPT] = modify_elements(
        P[ADOPT],
        states_num,
        2 * states_num,
        ordered_list,
        [index_column_current] * states_num,
        [1 - alpha] * states_num,
    )
    # probability under action override.
    # only works for a > h.
    # I am wrong at the first attempt. PLZ remember it !!!
    for a in range(0, rounds):
        for h in range(0, rounds):
            index_row_begin = get_index(a, h, IRRELEVANT, rounds, fork_states_num)
            index_row_end = index_row_begin + fork_states_num

            adversary_height, honest_height = 1, 0
            index_column_current = get_index(
                adversary_height, honest_height, IRRELEVANT, rounds, fork_states_num
            )

            # reward of adopt aciton.
            H[ADOPT] = modify_elements(
                H[ADOPT],
                cursor_H[ADOPT],
                cursor_H[ADOPT] + fork_states_num,
                [i for i in range(index_row_begin, index_row_end)],
                [index_column_current] * fork_states_num,
                [h] * fork_states_num,
            )
            cursor_H[ADOPT] += 3

            adversary_height, honest_height = 0, 1
            index_column_current = get_index(
                adversary_height, honest_height, IRRELEVANT, rounds, fork_states_num
            )

            H[ADOPT] = modify_elements(
                H[ADOPT],
                cursor_H[ADOPT],
                cursor_H[ADOPT] + fork_states_num,
                [i for i in range(index_row_begin, index_row_end)],
                [index_column_current] * fork_states_num,
                [h] * fork_states_num,
            )
            cursor_H[ADOPT] += 3

    for a in range(rounds - 1):
        for h in range(rounds - 1):
            # get row.
            index_row_begin = get_index(a, h, IRRELEVANT, rounds, fork_states_num)
            index_row_end = index_row_begin + fork_states_num
            values_zero = [0] * fork_states_num
            index_row_interval = [i for i in range(index_row_begin, index_row_end)]
            if a > h:
                # clear diagonal.
                P[OVERRIDE]["values"][index_row_begin:index_row_end] = values_zero
                H[OVERRIDE]["values"][index_row_begin:index_row_end] = values_zero

                # with probablity alpha to ((a - h), 0, 0).
                index_column_current = get_index(
                    (a - h), 0, IRRELEVANT, rounds, fork_states_num
                )
                P[OVERRIDE] = modify_elements(
                    P[OVERRIDE],
                    cursor_P[OVERRIDE],
                    cursor_P[OVERRIDE] + fork_states_num,
                    index_row_interval,
                    [index_column_current] * fork_states_num,
                    [alpha] * fork_states_num,
                )

                # reward of above action.
                A[OVERRIDE] = modify_elements(
                    A[OVERRIDE],
                    cursor_A[OVERRIDE],
                    cursor_A[OVERRIDE] + fork_states_num,
                    index_row_interval,
                    [index_column_current] * fork_states_num,
                    [h + 1] * fork_states_num,
                )

                cursor_P[OVERRIDE] += fork_states_num
                cursor_A[OVERRIDE] += fork_states_num

                # with probablity alpha to ((a - h s- 1), 1, 1).
                index_column_current = get_index(
                    (a - h - 1), 1, RELEVANT, rounds, fork_states_num
                )
                P[OVERRIDE] = modify_elements(
                    P[OVERRIDE],
                    cursor_P[OVERRIDE],
                    cursor_P[OVERRIDE] + fork_states_num,
                    index_row_interval,
                    [index_column_current] * fork_states_num,
                    [1 - alpha] * fork_states_num,
                )

                A[OVERRIDE] = modify_elements(
                    A[OVERRIDE],
                    cursor_A[OVERRIDE],
                    cursor_A[OVERRIDE] + fork_states_num,
                    index_row_interval,
                    [index_column_current] * fork_states_num,
                    [h + 1] * fork_states_num,
                )
                cursor_P[OVERRIDE] += fork_states_num
                cursor_A[OVERRIDE] += fork_states_num

            # probability under action wait.
            # clear diagonal.
            P[WAIT]["values"][index_row_begin : index_row_end - 1] = [0, 0]
            H[WAIT]["values"][index_row_begin : index_row_end - 1] = [0, 0]

            # WAIT:IRRELEVANT
            P[WAIT] = modify_elements(
                P[WAIT],
                cursor_P[WAIT],
                cursor_P[WAIT] + 2,
                [index_row_begin] * 2,
                [
                    get_index(a + 1, h, IRRELEVANT, rounds, fork_states_num),
                    get_index(a, h + 1, RELEVANT, rounds, fork_states_num),
                ],
                [alpha, 1 - alpha],
            )
            cursor_P[WAIT] += 2

            # WAIT:RELEVANT
            P[WAIT] = modify_elements(
                P[WAIT],
                cursor_P[WAIT],
                cursor_P[WAIT] + 2,
                [index_row_begin + 1] * 2,
                [
                    get_index(a + 1, h, IRRELEVANT, rounds, fork_states_num),
                    get_index(a, h + 1, RELEVANT, rounds, fork_states_num),
                ],
                [alpha, 1 - alpha],
            )
            cursor_P[WAIT] += 2

            # WAIT:ACTIVE
            if a >= h:
                P[WAIT]["values"][index_row_begin + 2] = 0
                H[WAIT]["values"][index_row_begin + 2] = 0

                P[WAIT] = modify_elements(
                    P[WAIT],
                    cursor_P[WAIT],
                    cursor_P[WAIT] + 3,
                    [index_row_begin + 2] * 3,
                    [
                        get_index(a + 1, h, ACTIVE, rounds, fork_states_num),
                        get_index(a - h, 1, RELEVANT, rounds, fork_states_num),
                        get_index(a, h + 1, RELEVANT, rounds, fork_states_num),
                    ],
                    [
                        alpha,
                        gamma * (1 - alpha),
                        (1 - gamma) * (1 - alpha),
                    ],
                )
                A[WAIT]["rows"][cursor_A[WAIT]] = index_row_begin + 2
                A[WAIT]["cols"][cursor_A[WAIT]] = get_index(
                    a - h, 1, RELEVANT, rounds, fork_states_num
                )
                A[WAIT]["values"][cursor_A[WAIT]] = h

                cursor_P[WAIT] += 3
                cursor_A[WAIT] += 1

            # probability under action match.
            if a >= h:
                # clear diagonal.
                P[MATCH]["values"][index_row_begin + 1] = 0
                H[MATCH]["values"][index_row_begin + 1] = 0

                # a, h, relevant.
                P[MATCH] = modify_elements(
                    P[MATCH],
                    cursor_P[MATCH],
                    cursor_P[MATCH] + 3,
                    [index_row_begin + 1] * 3,
                    [
                        get_index(a + 1, h, ACTIVE, rounds, fork_states_num),
                        get_index(a - h, 1, RELEVANT, rounds, fork_states_num),
                        get_index(a, h + 1, RELEVANT, rounds, fork_states_num),
                    ],
                    [
                        alpha,
                        gamma * (1 - alpha),
                        (1 - gamma) * (1 - alpha),
                    ],
                )
                A[MATCH]["rows"][cursor_A[MATCH]] = index_row_begin + 1
                A[MATCH]["cols"][cursor_A[MATCH]] = get_index(
                    a - h, 1, RELEVANT, rounds, fork_states_num
                )
                A[MATCH]["values"][cursor_A[MATCH]] = h
                cursor_P[MATCH] += 3
                cursor_A[MATCH] += 1

    for action in [OVERRIDE, WAIT, MATCH]:
        for key in P[action].keys():
            P[action][key] = np.delete(
                P[action][key], [i for i in range(cursor_P[action], states_num * 5)]
            )
        for key in A[action].keys():
            A[action][key] = np.delete(
                A[action][key], [i for i in range(cursor_A[action], states_num * 5)]
            )
        for key in H[action].keys():
            H[action][key] = np.delete(
                H[action][key], [i for i in range(cursor_H[action], states_num * 5)]
            )
    # get the result.
    P = convert_data_to_sparse_matrix(P, states_num)
    A = convert_data_to_sparse_matrix(A, states_num)
    H = convert_data_to_sparse_matrix(H, states_num)
    return P, A, H


if __name__ == "__main__":
    epsilon = 0.0001
    action_num, fork_states_num = 4, 3
    gamma = 0
    for alpha in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        # for alpha in [0.1]:
        # if alpha <= 0.4:
        #     rounds = 80 - 1
        # else:
        #     rounds = 160 - 1
        rounds = 95
        states_num = rounds * rounds * 3
        P, A, H = generate_matrixs(states_num, rounds, fork_states_num, alpha, gamma)
        low, high = 0, 1
        # UNDERPAYING
        while high - low > epsilon / 8:
            rho = (low + high) / 2
            R = [None] * 4
            # generate Reward with different rho.
            for action in [ADOPT, OVERRIDE, WAIT, MATCH]:
                R[action] = (1 - rho) * A[action] - rho * H[action]
            rvi = mdptoolbox.mdp.RelativeValueIteration(P, R, epsilon=epsilon / 8)
            rvi.run()

            if rvi.average_reward > 0:
                low = rho
            else:
                high = rho
        print("low bound: alpha: {}, gamma: {}, rho: {}".format(alpha, gamma, rho))

        low_bound = rho - epsilon
        rho_prime = max(low - epsilon / 4, 0)
        A, H = adjust_reward_with_overpaying(A, H, alpha, rho)
        # generate Reward with different rho.
        for action in [ADOPT, OVERRIDE, WAIT, MATCH]:
            R[action] = (1 - rho_prime) * A[action] - rho_prime * H[action]
        rvi = mdptoolbox.mdp.RelativeValueIteration(P, R, epsilon=epsilon / 8)
        rvi.run()
        print(
            "upper bound: alpha: {}, gamma: {}, rho: {}".format(
                alpha, gamma, rho_prime + 2 * (rvi.average_reward + epsilon)
            )
        )

        # # OVERPAYING
        # high = min(rho + 0.1, 1)
        # low = rho
        # while high - low > epsilon / 8:
        #     rho = (low + high) / 2
        #     R = [None] * 4
        #     # revise A and H.
        #     A, H = adjust_reward_with_overpaying(A, H, alpha, rho)
        #     # generate Reward with different rho.
        #     for action in [ADOPT, OVERRIDE, WAIT, MATCH]:
        #         R[action] = (1 - rho) * A[action] - rho * H[action]
        #     rvi = mdptoolbox.mdp.RelativeValueIteration(P, R, epsilon=epsilon / 8)
        #     rvi.run()
        #     if rvi.average_reward > 0:
        #         low = rho
        #     else:
        #         high = rho
        # print("upper bound: alpha: {}, gamma: {}, rho: {}".format(alpha, gamma, rho))
