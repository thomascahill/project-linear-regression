# test 
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self):
        #set random seed
        np.random.seed(42)

        #data - features & targets
        self.features = np.random.randint(500, 3500, size=100)
        noise = np.random.normal(0, 25, size=100)
        self.targets = 50 + 0.1 * self.features + noise

        #scale the features to accelerate convergence and improve the model
        self.mean = np.mean(self.features)
        self.std = np.std(self.features)
        self.scaled_features = (self.features - self.mean) / self.std

        # initialize parameters & learning rate
        self.w, self.b = 0.0, 0.0
        self.alpha = 0.01
        self.losses = []

    # model
    def model(self, x):
        return (self.w * x) + self.b

    # loss function
    def mean_squared_error(self, y, y_hat):
        return np.mean((y - y_hat) ** 2)

    # gradients
    def compute_gradients(self, x, y, y_hat):
        n = len(y)
        dJ_dw = (-2/n) * np.sum(x * (y - y_hat))
        dJ_db = (-2/n) * np.sum(y - y_hat)
        return dJ_dw, dJ_db

    # perform gradient descent
    def train(self, epochs=1000):
        for epoch in range(epochs):
            y_hat = self.model(self.scaled_features)
            loss = self.mean_squared_error(self.targets, y_hat)
            self.losses.append(loss)
            dJ_dw, dJ_db = self.compute_gradients(self.scaled_features, self.targets, y_hat)
            self.w -= self.alpha * dJ_dw
            self.b -= self.alpha * dJ_db

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss}")


        print(f"Final model: y = {self.w:.4f}x + {self.b:.4f}")

    
    def plot(self):
        # First Figure: Scatter + Regression Line
        plt.figure()
        plt.scatter(self.features, self.targets, color='blue', alpha=0.6)
        x_vals = np.linspace((min(self.features), max(self.features)), 100)
        x_vals_scaled = (x_vals - self.mean) / self.std
        y_vals = self.model(x_vals_scaled)
        plt.plot(x_vals, y_vals, color='red', label = 'fitted line')
        plt.title("Regression Model")
        plt.xlabel("Size")
        plt.ylabel("Price")
        plt.grid(True)

        # Second Figure: loss over epochs
        plt.figure()
        epoch_vals = np.linspace(0, 1000, 1000)
        plt.plot(epoch_vals, self.losses, color='blue')
        plt.title("Epoch Vs. Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        #display plots
        plt.show()

# linear_regression = LinearRegression()

# print(linear_regression.features)
# print(linear_regression.scaled_features)
# print(linear_regression.targets)
# linear_regression.train()
# linear_regression.plot()

