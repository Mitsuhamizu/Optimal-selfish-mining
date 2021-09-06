from re import S

import numpy as np
from scipy.sparse import coo_matrix as sparse_coo


class Matrix:
    # init a list of indexer for fast locate the index of a state.
    def __init__(self, length):
        self.row = np.zeros([length])
        self.col = np.zeros([length])
        self.p = np.zeros([length])
        self.a = np.zeros([length])
        self.h = np.zeros([length])
        self.cursor = 0

    def add_element(self, row, col, p, a, h):
        self.row[self.cursor] = row
        self.col[self.cursor] = col
        self.p[self.cursor] = p
        self.a[self.cursor] = a
        self.h[self.cursor] = h
        self.cursor += 1

    def transfer_to_sparse(self, states_num):
        # self.delete_element_from_cursor()
        P = sparse_coo(
            (self.p, (self.row, self.col)),
            shape=[states_num, states_num],
        ).tocsr()
        A = sparse_coo(
            (self.a, (self.row, self.col)),
            shape=[states_num, states_num],
        ).tocsr()
        H = sparse_coo(
            (self.h, (self.row, self.col)),
            shape=[states_num, states_num],
        ).tocsr()
        return P, A, H

    def delete_element_from_cursor(self):
        self.row = self.row[: self.cursor]
        self.col = self.col[: self.cursor]
        self.p = self.p[: self.cursor]
        self.a = self.a[: self.cursor]
        self.h = self.h[: self.cursor]
