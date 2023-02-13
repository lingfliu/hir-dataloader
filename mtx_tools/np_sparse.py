import time

import numpy as np
import scipy as sp

'''适用于numpy稀疏矩阵乘法'''
class NpSparseMat:

    """
    for extremely large sparse matrix, use from_raw instead
    """
    @classmethod
    def from_npmat(cls, np_mat):
        val = []
        row = []
        col = []
        values = np_mat.nonzero()
        for i, v in enumerate(values[0]):
            val.append(np_mat[values[0][i], values[1][i]])
            row.append(values[0][i])
            col.append(values[1][i])

        row = np.array(row)
        col = np.array(col)
        val = np.array(val)
        sorted_idx = row.argsort()

        mat = NpSparseMat()
        mat.val = val[sorted_idx]
        mat.row = row
        mat.col = col[sorted_idx]
        mat.row_len = np_mat.shape[0]
        mat.col_len = np_mat.shape[1]

        return mat

    @classmethod
    def from_raw(cls, row_len, col_len, row, col, val):
        mat = NpSparseMat()
        sorted_idx = row.argsort()

        mat.val = np.array(val[sorted_idx])
        mat.row = np.array(row)
        mat.col = np.array(col[sorted_idx])
        mat.row_len = row_len
        mat.col_len = col_len

        return mat


    def to_npmat(self):
        res = np.zeros((self.row_len, self.col_len))
        for i in range(self.val.shape[0]):
            res[self.row[i], self.col[i]] = self.val[i]
        return res

    """
    remove zero elements
    """
    def squeeze(self):
        idx = self.val != 0
        self.val = self.val[idx]
        self.row = self.row[idx]
        self.col = self.col[idx]

    # compute self*np_mat
    def multiply(self, np_mat):
        row_len2 = np_mat.shape[0]
        col_len2 = np_mat.shape[1]

        uni_row = np.unique(self.row)

        res = np.zeros((len(uni_row), col_len2))
        col = np.linspace(0, col_len2-1, col_len2, dtype=np.int32)
        concat_col = np.tile(col, len(uni_row))
        concat_row = np.repeat(uni_row, col_len2)

        cnt = 0
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                idx = self.row == uni_row[i]
                if np.any(idx):
                    res[i,j] = np.sum(self.val[idx] * np_mat[self.col[idx], j])
                else:
                    continue

        mat = NpSparseMat.from_raw(self.row_len, col_len2, concat_row, concat_col, res.reshape(-1))
        mat.squeeze()
        return mat

    """
    compute np_mat*self
    """
    def multiply_post(self, np_mat):
        row_len2 = np_mat.shape[0]
        col_len2 = np_mat.shape[1]

        uni_col = np.unique(self.col)

        res = np.zeros((row_len2, len(uni_col)))
        row = np.linspace(0, row_len2-1, row_len2, dtype=np.int32)
        concat_row = np.tile(row, len(uni_col))
        concat_col = np.repeat(uni_col, row_len2)

        cnt = 0
        for i in range(res.shape[1]):
            for j in range(res.shape[0]):
                idx = self.col == uni_col[i]
                if np.any(idx):
                    res[j,i] = np.sum(self.val[idx] * np_mat[i,self.row[idx]])
                else:
                    continue

        mat = NpSparseMat.from_raw(row_len2, self.col_len, concat_row, concat_col, res.reshape(-1))
        mat.squeeze()
        return mat

    """
    convert to scipy sparse matrix
    使用sp.sparse.linalg进行其他矩阵运算
    """
    def to_spmat(self):
        mat = self.to_npmat()
        return sp.sparse.csr_matrix(mat)

    """TODO: 直接从scipy sparse matrix创建"""
    @classmethod
    def from_spmat(self, sp_mat):
        mat = sp_mat.toarray()
        return NpSparseMat.from_npmat(mat)

#
# if __name__ == '__main__':
#
#     tic = time.time()
#     a = np.zeros((50000,50000))
#     a[0, 0] = 1
#     a[1, 1] = 1
#     a[2, 2] = 1
#     a[3, 3] = 1
#     a[300, 1] = 1
#     a[1524, 1524] = 1
#     a[2524, 2524] = 1
#     tac = time.time()
#     print('create npmat consume:', tac-tic)
#
#     tic = time.time()
#     sa = NpSparseMat.from_npmat(a)
#     tac = time.time()
#     print('create sparse npmat consume:', tac-tic)
#
#     tic = time.time()
#     b = sa.multiply(a)
#     tac = time.time()
#     print('consume:', tac-tic)
#     print(b.col)
#     print(b.row)
#     print(b.val)
#
#     tic = time.time()
#     b = sa.multiply_post(a)
#     tac = time.time()
#     print('consume:', tac-tic)
#     print(b.col)
#     print(b.row)
#     print(b.val)
#
#     tic = time.time()
#     print(b.to_npmat())
#
#     tac = time.time()
#     print('to npmat consume:', tac-tic)
