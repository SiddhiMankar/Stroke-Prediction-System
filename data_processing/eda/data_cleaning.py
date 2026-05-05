"""
Data Cleaning Script
Purpose: Drop unnecessary columns, handle missing values, investigate BMI outliers
"""

import pandas as pd
import numpy as np

# ===============================
# CONFIGURATION
# ===============================

DATA_PATH = r"C:\Projects\Stroke-awareness-TY-mini-project\data\stroke_cleaned.csv"
OUTPUT_PATH = r"C:\Projects\Stroke-awareness-TY-mini-project\data\stroke_cleaned_final.csv"

# Columns to drop
COLUMNS_TO_DROP = ['Unnamed: 0', 'id']

# ===============================
# 1. LOAD AND INSPECT DATA
# ===============================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nOriginal shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# ===============================
# 2. DROP UNNECESSARY COLUMNS
# ===============================

print("\n" + "=" * 60)
print("DROPPING UNNECESSARY COLUMNS")
print("=" * 60)

# Check which columns exist before dropping
existing_cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
print(f"\nColumns to drop: {existing_cols_to_drop}")

if existing_cols_to_drop:
    df = df.drop(columns=existing_cols_to_drop)
    print(f"Dropped {len(existing_cols_to_drop)} columns")
else:
    print("No columns to drop (already removed)")

print(f"New shape: {df.shape}")

# ===============================
# 3. INVESTIGATE MISSING VALUES
# ===============================

print("\n" + "=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)

missing = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_summary = pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Percentage': missing_percent
})

print("\nColumns with missing values:")
print(missing_summary[missing_summary['Missing_Count'] > 0])

# ===============================
# 4. INVESTIGATE BMI OUTLIERS
# ===============================

print("\n" + "=" * 60)
print("BMI OUTLIERS INVESTIGATION")
print("=" * 60)

if 'bmi' in df.columns:
    print("\nBMI Statistics:")
    print(df['bmi'].describe())
    
    bmi_over_60 = (df['bmi'] > 60).sum()
    bmi_over_70 = (df['bmi'] > 70).sum()
    
    print(f"\nBMI > 60: {bmi_over_60} ({bmi_over_60/len(df)*100:.2f}%)")
    print(f"BMI > 70: {bmi_over_70} ({bmi_over_70/len(df)*100:.2f}%)")
    
    if bmi_over_60 > 0:
        print("\nExtreme BMI values:")
        print(df[df['bmi'] > 60]['bmi'].sort_values(ascending=False))
else:
    print("No 'bmi' column found in dataset")

# ===============================
# 5. HANDLE MISSING VALUES
# ===============================

print("\n" + "=" * 60)
print("HANDLING MISSING VALUES")
print("=" * 60)

# Option A: Recreate age_group and bmi_category from base columns
if 'age_group' in df.columns and df['age_group'].isnull().sum() > 0:
    if 'age' in df.columns:
        print("\nRecreating age_group from age column...")
        # Define age groups
        df['age_group'] = pd.cut(df['age'], 
                                  bins=[0, 18, 35, 50, 65, 100],
                                  labels=['0-18', '19-35', '36-50', '51-65', '65+'],
                                  include_lowest=True)
        print(f"Recreated age_group. Missing values: {df['age_group'].isnull().sum()}")
    else:
        print("\nFilling age_group with mode...")
        mode_val = df['age_group'].mode()[0]
        df['age_group'].fillna(mode_val, inplace=True)
        print(f"Filled with mode: {mode_val}")

if 'bmi_category' in df.columns and df['bmi_category'].isnull().sum() > 0:
    if 'bmi' in df.columns:
        print("\nRecreating bmi_category from bmi column...")
        # Define BMI categories
        df['bmi_category'] = pd.cut(df['bmi'],
                                     bins=[0, 18.5, 25, 30, 100],
                                     labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
                                     include_lowest=True)
        print(f"Recreated bmi_category. Missing values: {df['bmi_category'].isnull().sum()}")
    else:
        print("\nFilling bmi_category with mode...")
        mode_val = df['bmi_category'].mode()[0]
        df['bmi_category'].fillna(mode_val, inplace=True)
        print(f"Filled with mode: {mode_val}")

# Handle other missing values if any
remaining_missing = df.isnull().sum()
if remaining_missing.sum() > 0:
    print("\nRemaining missing values:")
    print(remaining_missing[remaining_missing > 0])

# ===============================
# 6. CAP BMI OUTLIERS
# ===============================

print("\n" + "=" * 60)
print("CAPPING BMI OUTLIERS")
print("=" * 60)

if 'bmi' in df.columns:
    bmi_over_60 = (df['bmi'] > 60).sum()
    
    if bmi_over_60 > 0 and bmi_over_60 < len(df) * 0.01:  # Less than 1% of data
        print(f"\nCapping {bmi_over_60} BMI values > 60 to 60")
        df.loc[df['bmi'] > 60, 'bmi'] = 60
        print("BMI values capped successfully")
        
        # Update bmi_category if it exists
        if 'bmi_category' in df.columns:
            print("Updating bmi_category for capped values...")
            df.loc[df['bmi'] > 30, 'bmi_category'] = 'Obese'
    else:
        print(f"\nNo capping needed (outliers: {bmi_over_60})")

# ===============================
# 7. FINAL SUMMARY
# ===============================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"\nFinal shape: {df.shape}")
print(f"\nFinal columns: {df.columns.tolist()}")

missing_final = df.isnull().sum()
print(f"\nTotal missing values: {missing_final.sum()}")
if missing_final.sum() > 0:
    print("\nMissing values by column:")
    print(missing_final[missing_final > 0])

# ===============================
# 8. SAVE CLEANED DATA
# ===============================

print("\n" + "=" * 60)
print("SAVING CLEANED DATA")
print("=" * 60)

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nCleaned data saved to: {OUTPUT_PATH}")

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)
