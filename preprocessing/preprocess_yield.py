import pandas as pd
import numpy as np

def prepare_crop_yield_data(csv_path):
    print("Executing professional preprocessing pipeline for Crop Yield...")
    
    # 1. I/O Operations via Pandas
    df = pd.read_csv(csv_path)
    
    # FILTER STEP: Extract only rows where Crop is Rice
    rice_df = df[df['Crop'] == 'Rice'].copy()
    
    # Extract independent variables from the filtered subset
    X_raw = rice_df.drop(columns=['Crop', 'Yield']).to_numpy(dtype=np.float64)
    # We take the natural log of the yield vector to neutralize the 120,000 outlier
    y_raw = np.log(rice_df['Yield'].to_numpy(dtype=np.float64).reshape(-1, 1))
    
    n_samples, n_features = X_raw.shape
    print(f"Loaded rice sub-matrix. Features: {n_features}, Samples: {n_samples}")
    
    # 2. Manual Reproducible Index Shuffling
    np.random.seed(42)  # Strict seed for pipeline reproducibility
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_idx = int(0.8 * n_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train_raw = X_raw[train_indices]
    X_test_raw = X_raw[test_indices]
    y_train = y_raw[train_indices]
    y_test = y_raw[test_indices]
    
    # 3. Isolated Feature Scaling (Preventing Data Leakage)
    # axis=0 executes column-wise reductions across samples
    mean_train = np.mean(X_train_raw, axis=0, keepdims=True)
    std_train = np.std(X_train_raw, axis=0, keepdims=True)
    
    # Numerical stability patch
    epsilon = 1e-8
    std_train = np.where(std_train == 0, epsilon, std_train)
    
    # Broadcast subtraction and division
    X_train_scaled = (X_train_raw - mean_train) / std_train
    X_test_scaled = (X_test_raw - mean_train) / std_train
    
    # 4. Design Matrix Augmentation (Prepend Column of 1s for Bias)
    ones_train = np.ones((X_train_scaled.shape[0], 1), dtype=np.float64)
    ones_test = np.ones((X_test_scaled.shape[0], 1), dtype=np.float64)
    
    X_train = np.hstack((ones_train, X_train_scaled))
    X_test = np.hstack((ones_test, X_test_scaled))
    
    # 5. Pipeline Integrity Checks
    print("\n--- Pipeline Matrix Diagnostics ---")
    print(f"X_train Shape (with Bias): {X_train.shape} -> Expected: (80, 8)")
    print(f"y_train Shape:             {y_train.shape} -> Expected: (80, 1)")
    print(f"X_test Shape (with Bias):  {X_test.shape} -> Expected: (20, 8)")
    print(f"y_test Shape:              {y_test.shape} -> Expected: (20, 1)")
    
    # Sanity check mean/std of scaled training data (should be ~0 and ~1)
    print(f"Scaled Train Mean (First 3 features): {X_train_scaled[:, :3].mean(axis=0)}")
    print(f"Scaled Train Std (First 3 features):  {X_train_scaled[:, :3].std(axis=0)}")
    
    # Save processed arrays for down-stream modeling files
    np.save('processed/yield/X_train_yield.npy', X_train)
    np.save('processed/yield/X_test_yield.npy', X_test)
    np.save('processed/yield/y_train_yield.npy', y_train)
    np.save('processed/yield/y_test_yield.npy', y_test)
    print("\nPreprocessed arrays successfully saved to disk.")

if __name__ == "__main__":
    # Replace with your actual local filename if different
    prepare_crop_yield_data("data/crop_yield.csv")