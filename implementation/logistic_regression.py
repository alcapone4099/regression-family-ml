import numpy as np
import matplotlib.pyplot as plt

class ManualLogisticRegression:
    def __init__(self, alpha=0.01, epochs=1000):
        """
        alpha: Learning rate for step updates
        epochs: Total iterations through the design matrix
        """
        self.alpha = alpha
        self.epochs = epochs
        self.w = None
        self.loss_history = []

    def _sigmoid(self, z):
        """
        Applies the non-linear sigmoid activation function with clipping protection.
        """
        z_stable = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z_stable))

    def fit(self, X, y):
        """
        Maximizes the Bernoulli log-likelihood via Gradient Descent.
        """
        n_samples, n_features = X.shape
        # Initialize weight vector to zero (includes the intercept weight)
        self.w = np.zeros((n_features, 1))
        self.loss_history = []

        for epoch in range(self.epochs):
            # 1. Forward Pass: Compute raw logits and squashed probabilities
            z = X @ self.w
            y_pred = self._sigmoid(z)

            # 2. Compute Stable Binary Cross-Entropy Loss
            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
            loss = - (1.0 / n_samples) * np.sum(
                y * np.log(y_pred_clipped) + (1.0 - y) * np.log(1.0 - y_pred_clipped)
            )
            self.loss_history.append(loss)

            # 3. Vectorized Gradient Pass (Derived via our Matrix Calculus)
            gradient = (1.0 / n_samples) * (X.T @ (y_pred - y))

            # 4. Optimization Update Step
            self.w -= self.alpha * gradient

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return np.where(proba >= threshold, 1.0, 0.0)

    def evaluate(self, X, y_true):
        """
        Manually derives the confusion matrix elements to return key performance vectors.
        """
        y_pred = self.predict(X)
        
        # Explicit reshaping to prevent dimension broadcast errors
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

        # Boolean intersections for True Positives, True Negatives, False Positives, False Negatives
        tp = np.sum((y_true == 1.0) & (y_pred == 1.0))
        tn = np.sum((y_true == 0.0) & (y_pred == 0.0))
        fp = np.sum((y_true == 0.0) & (y_pred == 1.0))
        fn = np.sum((y_true == 1.0) & (y_pred == 0.0))

        # Precision, Recall, and Accuracy Equations
        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return accuracy, precision, recall, f1_score

    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(self.loss_history)), self.loss_history, color='teal', lw=2)
        plt.title("Logistic Regression BCE Loss Landscape Optimization")
        plt.xlabel("Epochs")
        plt.ylabel("Binary Cross-Entropy Cost")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()