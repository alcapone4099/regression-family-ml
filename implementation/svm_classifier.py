import numpy as np

class BinarySVM:
    """A single binary Support Vector Machine using Primal Subgradient Descent."""
    def __init__(self, C=1.0, alpha=0.001, epochs=1000):
        self.C = C
        self.alpha = alpha
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Check the hinge loss margin condition
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # Case 2: Correctly classified outside margin (only penalize weight length)
                    self.w -= self.alpha * (1.0 / self.epochs * self.w)
                else:
                    # Case 1: Margin violation / Misclassification
                    self.w -= self.alpha * (1.0 / self.epochs * self.w - self.C * y[idx] * x_i)
                    self.b -= self.alpha * (-self.C * y[idx])

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b


class MultiClassSVM:
    """Multi-class wrapper using a One-vs-Rest (OvR) strategy."""
    def __init__(self, C=1.0, alpha=0.001, epochs=1000):
        self.C = C
        self.alpha = alpha
        self.epochs = epochs
        self.classifiers = []
        self.unique_classes = None

    def fit(self, X, y):
        # Flatten y to ensure consistent boolean masking
        y = y.ravel()
        self.unique_classes = np.unique(y)
        self.classifiers = []

        for c in self.unique_classes:
            # Convert labels to binary targeting system: +1 for target class, -1 for all others
            binary_y = np.where(y == c, 1.0, -1.0)
            
            clf = BinarySVM(C=self.C, alpha=self.alpha, epochs=self.epochs)
            clf.fit(X, binary_y)
            self.classifiers.append(clf)

    def predict(self, X):
        # Gather scores across all classifiers: Shape (n_samples, n_classes)
        scores = np.column_stack([clf.decision_function(X) for clf in self.classifiers])
        # Assign the class with the highest confidence margin score
        best_class_indices = np.argmax(scores, axis=1)
        return self.unique_classes[best_class_indices].reshape(-1, 1)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        y_true = y_true.reshape(-1, 1)

        # Macro-Averaged Accuracy Calculation
        accuracy = np.mean(y_pred == y_true)
        
        # Calculate per-class metrics manually for a comprehensive review
        precision_list = []
        recall_list = []

        for c in self.unique_classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precision_list.append(prec)
            recall_list.append(rec)

        macro_precision = np.mean(precision_list)
        macro_recall = np.mean(recall_list)
        macro_f1 = 2.0 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0.0

        return accuracy, macro_precision, macro_recall, macro_f1