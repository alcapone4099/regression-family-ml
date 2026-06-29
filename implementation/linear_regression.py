import numpy as np
import matplotlib.pyplot as plt

class RidgeLinearRegression:
    def __init__(self, alpha=0.01, l2_reg=0.0, epochs=1000):
        """
        alpha: Learning rate for Gradient Descent
        l2_reg: Lambda (regularization penalty strength)
        epochs: Maximum iterations for Gradient Descent
        """
        self.alpha = alpha
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.w = None  # Weight vector
        self.loss_history = []

    def fit_closed_form(self, X, y):
        """
        Solves the regularized normal equations: w = (X^T X + n*lambda*I)^-1 X^T y
        """
        n_samples, n_features = X.shape
        I = np.eye(n_features)
        
        # Note: Professional practice typically does not regularize the intercept term (index 0).
        # For simplicity and tracking strict textbook definitions, we scale the full identity matrix.
        I[0, 0] = 0.0  # Do not penalize the bias weight!
        
        inside_inverse = X.T @ X + (n_samples * self.l2_reg * I)
        self.w = np.linalg.inv(inside_inverse) @ X.T @ y
        print("Closed-form weights computed successfully.")

    def fit_gradient_descent(self, X, y):
        """
        Iteratively minimizes loss using w = w - alpha * grad(J)
        """
        n_samples, n_features = X.shape
        # Initialize weights to zero
        self.w = np.zeros((n_features, 1))
        self.loss_history = []

        for epoch in range(self.epochs):
            predictions = X @ self.w
            error = predictions - y
            
            # Compute analytical gradient
            # Ensure the bias weight isn't artificially suppressed by regularization
            w_penalty = np.copy(self.w)
            w_penalty[0, 0] = 0.0
            
            gradient = (1.0 / n_samples) * (X.T @ error) + (self.l2_reg * w_penalty)
            
            # Step down the loss surface
            self.w -= self.alpha * gradient
            
            # Compute and record MSE cost for logging
            mse_loss = (1.0 / (2.0 * n_samples)) * np.sum(error ** 2) + (self.l2_reg / 2.0) * np.sum(w_penalty ** 2)
            self.loss_history.append(mse_loss)

    def predict(self, X):
        return X @ self.w

    def evaluate(self, X, y_true):
        """
        Calculates R^2 and RMSE manually using vector reductions
        """
        y_pred = self.predict(X)
        
        # Mean Squared Error & Root Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # Total Sum of Squares vs Residual Sum of Squares for R^2
        y_bar = np.mean(y_true)
        ss_tot = np.sum((y_true - y_bar) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)
        
        return r2, rmse

    def plot_loss(self):
        if not self.loss_history:
            print("No gradient descent history found to plot.")
            return
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(self.loss_history)), self.loss_history, color='crimson', lw=2)
        plt.title("Gradient Descent Optimization Landscape")
        plt.xlabel("Epochs")
        plt.ylabel("Cost J(w)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()