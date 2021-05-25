#!/usr/bin/python3

import numpy as np

# Calculate mean of array
def calc_mean(array):
    return np.mean(array)

A = np.random.rand(100)
print(calc_mean(A))
