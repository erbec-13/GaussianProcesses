#Author: Ethan Erb ece46

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import sys

def run_problem_1():
    # 1D input: Time -> Energy (with noise)
    X_train = np.array([[1.0], [3.0], [5.0]])
    y_train = np.array([2.0, 0.0, 1.0]) + np.random.normal(0, [0.1, 0.3, 0.2])

    # Define the squared exponential (RBF) kernel
    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0)

    # Instantiate a GP regressor
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1**2, n_restarts_optimizer=10)

    # Fit GP: This computes the marginal likelihood and optimizes kernel hyperparameters
    gp.fit(X_train, y_train)

    # Test points (time)
    X_test = np.linspace(0, 6, 100).reshape(-1, 1)

    # GP prediction -> posterior mean and covariance
    y_mean, y_std = gp.predict(X_test, return_std=True)

    # Plot prior, data, and posterior
    plt.figure(figsize=(10, 6))
    plt.title("Gaussian Process Regression - Energy Over Time")
    plt.fill_between(X_test.ravel(), y_mean - 1.96 * y_std, y_mean + 1.96 * y_std, 
                     alpha=0.2, label="95% Confidence Interval (Posterior)")
    plt.plot(X_test, y_mean, 'b-', label="Posterior Mean")
    plt.scatter(X_train, y_train, c='red', label="Training Data")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prediction at a specific test point
    test_input = np.array([[4.0]])
    pred_mean, pred_std = gp.predict(test_input, return_std=True)
    print(f"\nPrediction at time={test_input[0, 0]}:")
    print(f"  Mean = {pred_mean[0]:.4f}")
    print(f"  Std  = {pred_std[0]:.4f} (Variance = {pred_std[0]**2:.4f})")


def run_problem_2():
    # 2D input: [Temperature, Wind Speed] -> Real Feel Temp
    X_train = np.array([
        [25.0, 5.0],   # hot, light wind
        [10.0, 20.0],  # cold, high wind
        [15.0, 10.0]   # mild
    ])
    y_train = np.array([27.0, 3.0, 14.5])

    kernel = C(1.0, (1e-2, 1e2)) * RBF([10.0, 10.0])  # Different length scales for temp and wind

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1**2, n_restarts_optimizer=10)
    gp.fit(X_train, y_train)

    # Grid for visualization (fix wind at 10, vary temperature)
    temp = np.linspace(0, 30, 50)
    wind = np.linspace(0, 25, 50)
    T, W = np.meshgrid(temp, wind)
    X_test = np.vstack((T.ravel(), W.ravel())).T

    y_mean, y_std = gp.predict(X_test, return_std=True)

    # Plot mean prediction surface
    plt.figure(figsize=(10, 6))
    plt.contourf(T, W, y_mean.reshape(T.shape), cmap='viridis', levels=20)
    plt.colorbar(label='Predicted Real Feel Temp')
    plt.scatter(X_train[:, 0], X_train[:, 1], c='red', label="Training Data")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Wind Speed (km/h)")
    plt.title("Gaussian Process Regression - Real Feel Temperature")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prediction at a specific point
    test_input = np.array([[20.0, 15.0]])  # warm with moderate wind
    pred_mean, pred_std = gp.predict(test_input, return_std=True)
    print(f"\nPrediction at temp={test_input[0,0]}, wind={test_input[0,1]}:")
    print(f"  Mean = {pred_mean[0]:.4f}")
    print(f"  Std  = {pred_std[0]:.4f} (Variance = {pred_std[0]**2:.4f})")


def main():
    print("Choose a problem to run:")
    print("1: Energy over time (1D GP)")
    print("2: Real feel temperature (2D GP)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_problem_1()
    elif choice == "2":
        run_problem_2()
    else:
        print("Invalid input. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
