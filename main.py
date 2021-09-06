from functools import WRAPPER_UPDATES
from operator import pos
from os import fork, stat

import mdptoolbox
import numpy as np
from numpy.lib.function_base import delete
from numpy.lib.polynomial import _raise_power
from numpy.matrixlib.defmatrix import matrix
from scipy.sparse import coo_matrix as sparse_coo

from matirx import *

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
    return a, h, f


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
    A, H, P = (
        [None] * len(actions_lib),
        [None] * len(actions_lib),
        [None] * len(actions_lib),
    )
    # init default value for other three.
    matrixs = [None] * len(actions_lib)
    for action in actions_lib:
        matrixs[action] = Matrix(10 * states_num)
    ordered_list = np.arange(states_num)
    ones = np.ones([states_num])
    index_base_adversary_mine = get_index(1, 0, IRRELEVANT, rounds, fork_states_num)
    index_base_honest_mine = get_index(0, 1, IRRELEVANT, rounds, fork_states_num)
    # init other three actions.
    for action in [OVERRIDE, WAIT, MATCH]:
        matrixs[action].row[:states_num] = ordered_list
        matrixs[action].col[:states_num] = ordered_list
        matrixs[action].p[:states_num] = ones
        matrixs[action].h[:states_num] = ones * 1000000
        matrixs[action].cursor = states_num

    # probability under action adopt.
    matrixs[ADOPT].row[0:states_num] = ordered_list
    matrixs[ADOPT].col[0:states_num] = ones * index_base_adversary_mine
    matrixs[ADOPT].p[0:states_num] = ones * alpha

    matrixs[ADOPT].row[states_num : 2 * states_num] = ordered_list
    matrixs[ADOPT].col[states_num : 2 * states_num] = ones * index_base_honest_mine
    matrixs[ADOPT].p[states_num : 2 * states_num] = ones * (1 - alpha)
    matrixs[ADOPT].cursor = 2 * states_num

    # iter
    for index in range(states_num):
        a, h, f = get_state(index, rounds, fork_states_num)
        # reward of ADOPT
        matrixs[ADOPT].h[index] = h
        matrixs[ADOPT].h[index + states_num] = h
        if a < rounds - 1 and h < rounds - 1:
            # OVERRIDE
            if a > h:
                # clear the data
                matrixs[OVERRIDE].h[index] = 0
                matrixs[OVERRIDE].p[index] = 0

                matrixs[OVERRIDE].add_element(
                    index,
                    get_index(a - h, 0, IRRELEVANT, rounds, fork_states_num),
                    alpha,
                    h + 1,
                    0,
                )
                matrixs[OVERRIDE].add_element(
                    index,
                    get_index(a - h - 1, 1, RELEVANT, rounds, fork_states_num),
                    1 - alpha,
                    h + 1,
                    0,
                )
            # WAIT in IRRELEVANT and RELEVANT
            if f == IRRELEVANT or f == RELEVANT:
                matrixs[WAIT].h[index] = 0
                matrixs[WAIT].p[index] = 0
                matrixs[WAIT].add_element(
                    index,
                    get_index(a + 1, h, IRRELEVANT, rounds, fork_states_num),
                    alpha,
                    0,
                    0,
                )
                matrixs[WAIT].add_element(
                    index,
                    get_index(a, h + 1, RELEVANT, rounds, fork_states_num),
                    1 - alpha,
                    0,
                    0,
                )
            # WAIT in active
            if a >= h and f == ACTIVE:
                matrixs[WAIT].h[index] = 0
                matrixs[WAIT].p[index] = 0
                matrixs[WAIT].add_element(
                    index,
                    get_index(a + 1, h, ACTIVE, rounds, fork_states_num),
                    alpha,
                    0,
                    0,
                )
                matrixs[WAIT].add_element(
                    index,
                    get_index(a - h, 1, RELEVANT, rounds, fork_states_num),
                    gamma * (1 - alpha),
                    h,
                    0,
                )
                matrixs[WAIT].add_element(
                    index,
                    get_index(a, h + 1, RELEVANT, rounds, fork_states_num),
                    (1 - gamma) * (1 - alpha),
                    0,
                    0,
                )
            # MATCH in RELEVANT
            if a >= h and f == RELEVANT:
                matrixs[MATCH].h[index] = 0
                matrixs[MATCH].p[index] = 0
                matrixs[MATCH].add_element(
                    index,
                    get_index(a + 1, h, ACTIVE, rounds, fork_states_num),
                    alpha,
                    0,
                    0,
                )
                matrixs[MATCH].add_element(
                    index,
                    get_index(a - h, 1, RELEVANT, rounds, fork_states_num),
                    gamma * (1 - alpha),
                    h,
                    0,
                )
                matrixs[MATCH].add_element(
                    index,
                    get_index(a, h + 1, RELEVANT, rounds, fork_states_num),
                    (1 - gamma) * (1 - alpha),
                    0,
                    0,
                )
    # Finish the matrix generation, then try to transfer it to sparse metrixs.
    for action in actions_lib:
        P[action], A[action], H[action] = matrixs[action].transfer_to_sparse(states_num)
    return P, A, H


if __name__ == "__main__":
    epsilon = 0.0001
    action_num, fork_states_num = 4, 3
    for gamma in [0, 0.5, 1]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
            # for alpha in [0.1]:
            if alpha <= 0.4:
                max_fork_len = 80
            else:
                max_fork_len = 160
            rounds = max_fork_len + 1
            states_num = rounds * rounds * 3
            P, A, H = generate_matrixs(
                states_num, rounds, fork_states_num, alpha, gamma
            )
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
            print(
                "low bound: alpha: {}, gamma: {}, rho: {:.4f}".format(alpha, gamma, rho)
            )

        # low_bound = rho - epsilon
        # rho_prime = max(low - epsilon / 4, 0)
        # A, H = adjust_reward_with_overpaying(A, H, alpha, rho)
        # # generate Reward with different rho.
        # for action in [ADOPT, OVERRIDE, WAIT, MATCH]:
        #     R[action] = (1 - rho_prime) * A[action] - rho_prime * H[action]
        # rvi = mdptoolbox.mdp.RelativeValueIteration(P, R, epsilon=epsilon / 8)
        # rvi.run()
        # print(
        #     "upper bound: alpha: {}, gamma: {}, rho: {}".format(
        #         alpha, gamma, rho_prime + 2 * (rvi.average_reward + epsilon)
        #     )
        # )

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
        # print("upper bound: alpha: {}, gamma: {}, rho: {:.4f}".format(alpha, gamma, rho))
