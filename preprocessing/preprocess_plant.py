import pandas as pd
import numpy as np

def prepare_plant_disease_data(csv_path):
    print("Executing preprocessing pipeline for Plant Disease (Logistic Regression)...")
    
    # 1. Read raw dataset
    df = pd.read_csv(csv_path)
    X_raw = df[['mean_green_intensity', 'texture_contrast']].to_numpy(dtype=np.float64)
    y_raw = df['disease_status'].to_numpy(dtype=np.float64).reshape(-1, 1)
    
    # 2. Train/Test Split (80/20) with fixed seed
    np.random.seed(42)
    indices = np.arange(X_raw.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(0.8 * len(indices))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y_raw[train_idx], y_raw[test_idx]
    
    # 3. Z-Score Standardization (Train-isolated)
    mean_train = np.mean(X_train_raw, axis=0, keepdims=True)
    std_train = np.std(X_train_raw, axis=0, keepdims=True)
    
    X_train_scaled = (X_train_raw - mean_train) / (std_train + 1e-8)
    X_test_scaled = (X_test_raw - mean_train) / (std_train + 1e-8)
    
    # 4. Augment Design Matrix with Column of Ones (Intercept Trick)
    ones_train = np.ones((X_train_scaled.shape[0], 1))
    ones_test = np.ones((X_test_scaled.shape[0], 1))
    
    X_train = np.hstack((ones_train, X_train_scaled))
    X_test = np.hstack((ones_test, X_test_scaled))
    
    # Diagnostics
    print("\n--- Plant Disease Matrix Diagnostics ---")
    print(f"X_train Shape (with Bias): {X_train.shape} -> Expected: (800, 3)")
    print(f"y_train Shape:             {y_train.shape} -> Expected: (800, 1)")
    print(f"X_test Shape (with Bias):  {X_test.shape} -> Expected: (200, 3)")
    
    # Save to disk
    np.save('processed/plant/X_train_plant.npy', X_train)
    np.save('processed/plant/X_test_plant.npy', X_test)
    np.save('processed/plant/y_train_plant.npy', y_train)
    np.save('processed/plant/y_test_plant.npy', y_test)
    print("Plant disease arrays saved successfully.")

if __name__ == "__main__":
    prepare_plant_disease_data("data/plant_disease.csv")