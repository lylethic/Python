import numpy as np
import matplotlib.pyplot as plt

# # Data: Height (X) in cm and Weight (y) in kg
# X = np.array([147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183])
# y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# # Parameters (weights)
# theta_0 = 0  # Intercept
# theta_1 = 0  # Slope


# learning_rate = 0.0000001  # Reduced learning rate: toc do hoc
# epochs = 10000  # Increased number of iterations: lan lap


# # Number of data points
# m = len(X)

# # Gradient Descent Algorithm
# for _ in range(epochs):
# # Calculate the predicted y values
#     y_pred = theta_0 + theta_1 * X
   
#     # Compute gradients
#     d_theta_0 = (-2/m) * np.sum(y - y_pred)  # Gradient for theta_0 (intercept)
#     d_theta_1 = (-2/m) * np.sum((y - y_pred) * X)  # Gradient for theta_1 (slope)
   
#     # Update the parameters (theta_0 and theta_1)
#     theta_0 = theta_0 - learning_rate * d_theta_0
#     theta_1 = theta_1 - learning_rate * d_theta_1

# # After training, print the final parameters
# print(f"Final intercept (theta_0): {theta_0}")
# print(f"Final slope (theta_1): {theta_1}")

# # Predict y values using the optimized parameters
# y_pred = theta_0 + theta_1 * X

# # Plot the data points and the linear regression line
# plt.scatter(X, y, color='blue', label='Data points (Height vs Weight)')
# plt.plot(X, y_pred, color='red', label='Regression line (Gradient Descent)')
# plt.xlabel("Height (cm)")
# plt.ylabel("Weight (kg)")
# plt.title("Height vs Weight - Linear Regression (Gradient Descent)")
# plt.legend()
# plt.show()

##****************************************************************##
# Data: Height (X) in cm and Weight (y) in kg
X = np.array([147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183])
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# Feature scaling (normalization of X)
X_scaled = (X - np.mean(X)) / np.std(X)

# Initialize parameters (weights)
theta_0 = 0  # Intercept
theta_1 = 0  # Slope: do doc

# Hyperparameters
learning_rate = 0.01  # Learning rate (after scaling)
epochs = 20000  # Number of iterations

# Number of data points
m = len(X_scaled)

# Gradient Descent Algorithm
for _ in range(epochs):
    # Calculate the predicted y values
    y_pred = theta_0 + theta_1 * X_scaled
   
    # Compute gradients: tinh toan do doc
    d_theta_0 = (-2/m) * np.sum(y - y_pred)  # Gradient for theta_0 (intercept)
    d_theta_1 = (-2/m) * np.sum((y - y_pred) * X_scaled)  # Gradient for theta_1 (slope)
   
    # Update the parameters (theta_0 and theta_1)
    theta_0 = theta_0 - learning_rate * d_theta_0
    theta_1 = theta_1 - learning_rate * d_theta_1

# After training, print the final parameters
print(f"Final intercept (theta_0): {theta_0}")
print(f"Final slope (theta_1): {theta_1}")

# Predict y values using the optimized parameters
y_pred = theta_0 + theta_1 * X_scaled

# Plot the data points and the linear regression line
plt.scatter(X, y, color='blue', label='Data points (Height vs Weight)')
plt.plot(X, y_pred, color='red', label='Regression line (Gradient Descent)')
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Height vs Weight - Linear Regression (Gradient Descent)")
plt.legend()
plt.show()

