import pandas as pd
import numpy as np

# 1. Ingest data (Pandas is ONLY allowed for I/O)
df = pd.read_csv("Crop_Yield_Prediction.csv")

# 2. Separate features and target, dropping the categorical 'Crop' column
X_raw = df.drop(columns=['Crop', 'Yield']).to_numpy(dtype=np.float64)
y_raw = df['Yield'].to_numpy(dtype=np.float64).reshape(-1, 1)

print(f"Raw Features Shape: {X_raw.shape}")
print(f"Raw Target Shape: {y_raw.shape}")