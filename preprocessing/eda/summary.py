"""
Quick Summary of Data Cleaning Results
"""

import pandas as pd

CLEANED_PATH = r"C:\Projects\Stroke-awareness-TY-mini-project\data\stroke_cleaned_final.csv"

df = pd.read_csv(CLEANED_PATH)

print("CLEANED DATASET SUMMARY")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\nMissing Values: {df.isnull().sum().sum()}")

if 'bmi' in df.columns:
    print(f"\nBMI Stats:")
    print(f"  Min: {df['bmi'].min():.2f}")
    print(f"  Max: {df['bmi'].max():.2f}")
    print(f"  Mean: {df['bmi'].mean():.2f}")
    print(f"  Values > 60: {(df['bmi'] > 60).sum()}")
