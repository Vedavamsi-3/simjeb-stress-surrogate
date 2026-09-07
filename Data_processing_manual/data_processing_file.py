import pandas as pd
import numpy as np

# 1. Load the new raw dataset
df = pd.read_csv('raw_meta_data_file.csv')

# 2. Step 1: Remove domain errors (corrupted negative genus entries)
df_clean = df[df['genus'] >= 0].copy()

# 3. Step 2: Automatically pick ALL numerical feature columns (excluding 'id')
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in numeric_cols if col != 'id']

# 4. Step 3: Apply log(1 + x) only to skewed numerical features (|skew| > 1.0)
df_log = df_clean[features].copy()

for col in features:
    if abs(df_clean[col].skew()) > 1.0:
        df_log[col] = np.log1p(df_log[col])

# 5. Step 4: Calculate Z-scores across numerical features
z_scores = (df_log - df_log.mean()) / df_log.std()
has_outlier = (z_scores.abs() > 3.5).any(axis=1)

# 6. Step 5: Filter out outlier rows (keeps all original text/metadata columns intact)
df_final = df_clean[~has_outlier].reset_index(drop=True)
df_final.to_csv('cleaned_raw_meta_data_file.csv', index=False)

# Output summary
print(f"Original shape: {df.shape}")
print(f"Cleaned shape:  {df_final.shape}")
print(f"Numerical features evaluated: {features}\n")

print("--- Cleaned Dataset Preview (First 5 Rows) ---")
print(df_final.head(5))