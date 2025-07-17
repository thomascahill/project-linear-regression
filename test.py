#test
import numpy as np


a = np.array([1, 2]) # shape(2, )
b = np.array([3, 4]) # shape(2, )

A, B = np.meshgrid(a, b) # shape (2, 2)

C = A[..., np.newaxis]
print(C)