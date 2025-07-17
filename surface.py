import numpy as np
import matplotlib.pyplot as plt
from play import LinearRegression

# Prepare data
linear_regression = LinearRegression()
y = linear_regression.targets
x = linear_regression.scaled_features

# Manually run gradient descent to record (w, b, loss) at each step
w_history = []
b_history = []
loss_history = []

w = 0.0
b = 0.0
alpha = 0.01
n_epochs = 1000 # Match the epochs in play.py
for epoch in range(n_epochs):
    y_hat = w * x + b
    n = len(y)
    dJ_dw = (-2/n) * np.sum(x * (y - y_hat))
    dJ_db = (-2/n) * np.sum(y - y_hat)
    w -= alpha * dJ_dw
    b -= alpha * dJ_db
    loss = np.mean((y - y_hat) ** 2)
    w_history.append(w)
    b_history.append(b)
    loss_history.append(loss)

w_history = np.array(w_history)
b_history = np.array(b_history)
loss_history = np.array(loss_history)

# Create meshgrid for loss surface - adjust range to match optimization path
w_min, w_max = w_history.min(), w_history.max()
b_min, b_max = b_history.min(), b_history.max()

# Add some padding around the path
w_padding = (w_max - w_min) * 0.1
b_padding = (b_max - b_min) * 0.1
w_grid = np.linspace(w_min - w_padding, w_max + w_padding, 100)
b_grid = np.linspace(b_min - b_padding, b_max + b_padding, 100)
W, B = np.meshgrid(w_grid, b_grid)

# Compute loss surface
W_3d = W[..., np.newaxis]  # (100, 100, 1)
B_3d = B[..., np.newaxis]  # (100, 100, 1)
x_3d = x.reshape(1, 1, -1)  # (1, 1, n_samples)
y_3d = y.reshape(1, 1, -1)  # (1, 1, n_samples)
Y_hat = W_3d * x_3d + B_3d  # (100, 100, n_samples)
Z = np.mean((y_3d - Y_hat) ** 2, axis=2)  # (100, 100)

# Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(W, B, Z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)

# Plot the optimization trajectory
ax.plot(w_history, b_history, loss_history, 'r-', linewidth=2, label='Gradient Descent Path')
ax.scatter(w_history[0], b_history[0], loss_history[0], color='green', s=40, label='Start', zorder=5)
ax.scatter(w_history[-1], b_history[-1], loss_history[-1], color='red', s=40, label='Optimal', zorder=5)

ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('Loss (MSE)')
ax.set_title('Loss Surface with Gradient Descent Optimization Path')
ax.legend()
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()

print(f"Optimized w: {w_history[-1]:.4f}")
print(f"Optimized b: {b_history[-1]:.4f}")
print(f"Final loss: {loss_history[-1]:.4f}") 