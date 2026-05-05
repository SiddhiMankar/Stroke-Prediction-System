"""
Verify Data Cleaning Results
"""

import pandas as pd

ORIGINAL_PATH = r"C:\Projects\Stroke-awareness-TY-mini-project\data\stroke_cleaned.csv"
CLEANED_PATH = r"C:\Projects\Stroke-awareness-TY-mini-project\data\stroke_cleaned_final.csv"

print("=" * 60)
print("COMPARING ORIGINAL VS CLEANED DATA")
print("=" * 60)

# Load both datasets
df_original = pd.read_csv(ORIGINAL_PATH)
df_cleaned = pd.read_csv(CLEANED_PATH)

print(f"\nORIGINAL:")
print(f"  Shape: {df_original.shape}")
print(f"  Columns: {len(df_original.columns)}")
print(f"  Missing values: {df_original.isnull().sum().sum()}")

print(f"\nCLEANED:")
print(f"  Shape: {df_cleaned.shape}")
print(f"  Columns: {len(df_cleaned.columns)}")
print(f"  Missing values: {df_cleaned.isnull().sum().sum()}")

print(f"\nCHANGES:")
print(f"  Columns dropped: {len(df_original.columns) - len(df_cleaned.columns)}")
print(f"  Missing values removed: {df_original.isnull().sum().sum() - df_cleaned.isnull().sum().sum()}")

# Check dropped columns
dropped_cols = set(df_original.columns) - set(df_cleaned.columns)
if dropped_cols:
    print(f"\n  Dropped columns: {list(dropped_cols)}")

# BMI analysis
if 'bmi' in df_cleaned.columns:
    print(f"\nBMI ANALYSIS:")
    print(f"  Original BMI max: {df_original['bmi'].max():.2f}")
    print(f"  Cleaned BMI max: {df_cleaned['bmi'].max():.2f}")
    print(f"  Original BMI > 60: {(df_original['bmi'] > 60).sum()}")
    print(f"  Cleaned BMI > 60: {(df_cleaned['bmi'] > 60).sum()}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
