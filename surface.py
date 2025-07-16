import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from play import LinearRegression

# import parameters
linear_regression = LinearRegression()

y = linear_regression.targets
x = linear_regression.scaled_features

# create meshgrid
w = np.linspace(-100, 100, 100)
b = np.linspace(-100, 100, 100)
W, B = np.meshgrid(w,b)



# compute Y_hat for each (w, b) pair
# W and B are (100, 100), we want to compute predictions for each (w, b) across all x
# So we expand W and B to (100, 100, 1) and x to (1, 1, 100)
W_exp = W[:, :, np.newaxis] # shape: (100, 100, 1)
B_exp = B[:, :, np.newaxis] # shape: (100, 100, 1)
x_exp = x[np.newaxis, np.newaxis, :] # shape: (1, 1, 100)
y_exp = y[np.newaxis, np.newaxis, :] # shape: (1, 1, 100)

# Compute predictions: (100, 100, 100)
Y_hat = W_exp * x_exp + B_exp

# Compute loss for each (w, b) pair
Z = np.mean((y_exp - Y_hat) ** 2, axis=2)  # shape: (100, 100)


print(Z.shape)

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, Z, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
ax.set_title('Loss Surface')
plt.show()