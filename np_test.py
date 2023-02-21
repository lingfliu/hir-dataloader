import random
import time

import numpy as np
import scipy as sp
import torch


a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[1,0,0]])

print(a.shape)
print(b.shape)

# c = np.multiply(np.transpose(b), a)
c = b@a
print(c.shape)

print(a)
print(b)
print(c)