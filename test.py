#test
import numpy as np


a = np.array([1, 2]) # shape(2, )
b = np.array([3, 4]) # shape(2, )

A, B = np.meshgrid(a, b) # shape (2, 2)

print(A)
print(B)
print("\n")

A = A[:, :, np.newaxis]

print(A)
print(A.shape)