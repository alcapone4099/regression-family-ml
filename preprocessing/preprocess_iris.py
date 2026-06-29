import pandas as pd
import numpy as np

def prepare_iris_data(csv_path):
    print("Executing preprocessing pipeline for Iris (Multi-Class SVM)...")
    
    # 1. Read raw dataset
    df = pd.read_csv(csv_path)
    
    # Map string labels to numeric integers
    label_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    df['species_int'] = df['species'].map(label_mapping)
    
    X_raw = df.drop(columns=['species', 'species_int']).to_numpy(dtype=np.float64)
    y_raw = df['species_int'].to_numpy(dtype=np.int64)
    
    # 2. Train/Test Split (80/20)
    np.random.seed(42)
    indices = np.arange(X_raw.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(0.8 * len(indices))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train_int, y_test_int = y_raw[train_idx], y_raw[test_idx]
    
    # 3. Z-Score Standardization (NO INTERCEPT COLUMN ADDED!)
    mean_train = np.mean(X_train_raw, axis=0, keepdims=True)
    std_train = np.std(X_train_raw, axis=0, keepdims=True)
    
    X_train = (X_train_raw - mean_train) / (std_train + 1e-8)
    X_test = (X_test_raw - mean_train) / (std_train + 1e-8)
    
    # 4. Manual One-vs-Rest (OvR) Target Generation with SVM Geometry (-1 vs +1)
    n_classes = 3
    y_train_ovr = np.zeros((y_train_int.shape[0], n_classes))
    y_test_ovr = np.zeros((y_test_int.shape[0], n_classes))
    
    for c in range(n_classes):
        # If the sample equals current class c, label it +1, else -1
        y_train_ovr[:, c] = np.where(y_train_int == c, 1.0, -1.0)
        y_test_ovr[:, c] = np.where(y_test_int == c, 1.0, -1.0)
        
    # Diagnostics
    print("\n--- Iris SVM Matrix Diagnostics ---")
    print(f"X_train Shape (Strictly Features): {X_train.shape} -> Expected: (120, 4)")
    print(f"y_train_ovr Shape (3 Class Columns): {y_train_ovr.shape} -> Expected: (120, 3)")
    print(f"X_test Shape:                       {X_test.shape} -> Expected: (30, 4)")
    
    # Quick structural check of first row if it belongs to class 0
    print(f"Sample 0 target vector configuration: {y_train_ovr[0]}")
    
    # Save out to disk
    np.save('processed/iris/X_train_iris.npy', X_train)
    np.save('processed/iris/X_test_iris.npy', X_test)
    np.save('processed/iris/y_train_iris.npy', y_train_ovr)
    np.save('processed/iris/y_test_iris.npy', y_test_ovr)
    print("Iris SVM arrays saved successfully.")

if __name__ == "__main__":
    prepare_iris_data("data/iris.csv")