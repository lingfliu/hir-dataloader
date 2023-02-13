import random
import time

import numpy as np
import scipy as sp
import torch

a = np.zeros((10,10))
# a[1,1] = 1
# a[2, 2] = 2
# a[3, 3] = 3
# a[4, 3] = 3
for i in range(10):
    a[i,i] = 10
    # a[i, 9-i] = random.Random().random()
# a = np.array([[0,0,0,1], [0,0,1,0], [0,0,0,0], [0,0,0,0]])

# c = np.ones((10, 1))
# b = sp.sparse.csc_matrix(a)

# print(a)
# print(b)

# d = b.multiply(c)
# print(d)
# d = sp.sparse.linalg.inv(b)
# print(d)
#
# print(time.time())
# print(a*a)
# print(time.time())
#
# c = b
# print(time.time())
# d = b.multiply(c)
# d = b@b
# print(d.max())
# print(time.time())
#
# c = 1

# row = np.array([1.1, 2.1, 3.1, 0.1, 1, 2, 3, 0])
# col = np.array([1, 2, 3, 4, 5, 6, 7, 8])
#
# sorted_idx = row.argsort()
# print(row)
# print(col[sorted_idx])
#
# mat = np.array([[1,2,3], [4,5,6], [7,8,9]])
#
# a = mat.reshape(-1)
#
# b = 0


a = np.zeros((20000,20000))
for i in range(1000):
    a[int(random.random()*5000), int(random.random()*5000)] = 1
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

ma = torch.clone(torch.from_numpy(a)).detach().to(device)
mb = torch.clone(ma)

tic = time.time()
da = torch.mm(ma, mb)
tac = time.time()
print(da)
print(tac-tic)

ma = ma.to_sparse_coo()
tic = time.time()
da = torch.mm(ma, mb)
tac = time.time()
print(da)
print(tac-tic)
