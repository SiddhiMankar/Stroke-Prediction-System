"""
Comprehensive Exploratory Data Analysis (EDA) for Stroke Dataset
==================================================================

This script performs a systematic 8-phase analysis of the stroke dataset:
1. Target Understanding - Class imbalance analysis
2. Numeric Features - Age, glucose, BMI vs stroke
3. Binary Risk Factors - Hypertension, heart disease
4. Lifestyle Factors - Smoking, work type, residence
5. Demographics - Gender, marital status
6. Engineered Features - Age groups, BMI categories
7. Correlation Analysis - Numeric feature relationships
8. Advanced Insights - Pairplot (optional)

All categorical comparisons use PERCENTAGE-BASED plots to handle class imbalance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'data/stroke_cleaned_final.csv'
OUTPUT_DIR = Path('eda/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_LARGE = (12, 8)
FIGSIZE_HEATMAP = (10, 8)

# Column definitions
TARGET = 'stroke'
NUMERIC_COLS = ['age', 'avg_glucose_level', 'bmi']
CATEGORICAL_COLS = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                    'work_type', 'Residence_type', 'smoking_status', 
                    'age_group', 'bmi_category']

print("="*80)
print("COMPREHENSIVE EDA FOR STROKE DATASET")
print("="*80)

# Load data
print("\n📊 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"✓ Columns: {df.columns.tolist()}")

# ============================================================================
# PHASE 1: UNDERSTAND THE TARGET
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: UNDERSTAND THE TARGET")
print("="*80)

stroke_counts = df[TARGET].value_counts()
stroke_pct = df[TARGET].value_counts(normalize=True) * 100

print(f"\nStroke Distribution:")
print(f"  No Stroke (0): {stroke_counts[0]:,} ({stroke_pct[0]:.2f}%)")
print(f"  Stroke (1):    {stroke_counts[1]:,} ({stroke_pct[1]:.2f}%)")
print(f"\n💡 Insight: Severe class imbalance detected!")
print(f"   - {stroke_pct[0]:.1f}% vs {stroke_pct[1]:.1f}%")
print(f"   - This justifies why RECALL matters more than accuracy")
print(f"   - Accuracy would be misleading (can achieve ~95% by always predicting 'no stroke')")

# Plot 1: Stroke Distribution
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
bars = ax.bar([0, 1], stroke_counts.values, color=['#2ecc71', '#e74c3c'], 
              edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Stroke', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Stroke Cases\n(Severe Class Imbalance)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks([0, 1])
ax.set_xticklabels(['No Stroke', 'Stroke'], fontsize=12)

# Add count and percentage annotations
for i, (count, pct) in enumerate(zip(stroke_counts.values, stroke_pct.values)):
    ax.text(i, count + 50, f'{count:,}\n({pct:.2f}%)', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_stroke_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_stroke_distribution.png")

# ============================================================================
# PHASE 2: NUMERIC FEATURES VS STROKE
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: NUMERIC FEATURES VS STROKE")
print("="*80)

# Helper function for numeric feature analysis
def analyze_numeric_feature(feature_name, plot_num):
    """Analyze and plot numeric feature vs stroke"""
    print(f"\n📈 Analyzing {feature_name}...")
    
    # Statistics
    stats_no_stroke = df[df[TARGET] == 0][feature_name].describe()
    stats_stroke = df[df[TARGET] == 1][feature_name].describe()
    
    print(f"  No Stroke - Median: {stats_no_stroke['50%']:.2f}, Mean: {stats_no_stroke['mean']:.2f}")
    print(f"  Stroke    - Median: {stats_stroke['50%']:.2f}, Mean: {stats_stroke['mean']:.2f}")
    
    median_diff = stats_stroke['50%'] - stats_no_stroke['50%']
    print(f"  Median Difference: {median_diff:+.2f}")
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    bp = axes[0].boxplot([df[df[TARGET] == 0][feature_name].dropna(),
                           df[df[TARGET] == 1][feature_name].dropna()],
                          labels=['No Stroke', 'Stroke'],
                          patch_artist=True,
                          widths=0.6)
    
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0].set_ylabel(feature_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    axes[0].set_title(f'Boxplot: {feature_name.replace("_", " ").title()} vs Stroke', 
                      fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Violin plot
    parts = axes[1].violinplot([df[df[TARGET] == 0][feature_name].dropna(),
                                 df[df[TARGET] == 1][feature_name].dropna()],
                                positions=[0, 1],
                                showmeans=True,
                                showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['No Stroke', 'Stroke'])
    axes[1].set_ylabel(feature_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    axes[1].set_title(f'Violin Plot: {feature_name.replace("_", " ").title()} vs Stroke', 
                      fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{plot_num:02d}_{feature_name}_vs_stroke.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_num:02d}_{feature_name}_vs_stroke.png")
    
    # Insight
    if median_diff > 0:
        print(f"  💡 Insight: Stroke patients have HIGHER {feature_name} (median +{median_diff:.2f})")
        if abs(median_diff) > stats_no_stroke['std']:
            print(f"     → Strong predictor (difference > 1 std dev)")
    else:
        print(f"  💡 Insight: Weak relationship (median difference: {median_diff:.2f})")

# Analyze each numeric feature
analyze_numeric_feature('age', 2)
analyze_numeric_feature('avg_glucose_level', 3)
analyze_numeric_feature('bmi', 4)

# ============================================================================
# PHASE 3: BINARY RISK FACTORS VS STROKE
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: BINARY RISK FACTORS VS STROKE")
print("="*80)

def analyze_binary_risk_factor(feature_name, plot_num):
    """Analyze binary risk factor with percentage-based plot"""
    print(f"\n🏥 Analyzing {feature_name}...")
    
    # Calculate stroke percentage for each category
    stroke_pct = df.groupby(feature_name)[TARGET].agg(['sum', 'count'])
    stroke_pct['percentage'] = (stroke_pct['sum'] / stroke_pct['count']) * 100
    
    print(f"  Stroke Rate by {feature_name}:")
    for idx, row in stroke_pct.iterrows():
        print(f"    {feature_name}={idx}: {row['percentage']:.2f}% ({int(row['sum'])}/{int(row['count'])})")
    
    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars = ax.bar(stroke_pct.index.astype(str), stroke_pct['percentage'], 
                  color=['#3498db', '#e74c3c'], edgecolor='black', 
                  linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel(feature_name.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.set_ylabel('Stroke Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Stroke Percentage by {feature_name.replace("_", " ").title()}\n(Percentage-Based Analysis)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add percentage annotations
    for i, (idx, row) in enumerate(stroke_pct.iterrows()):
        ax.text(i, row['percentage'] + 0.3, f"{row['percentage']:.2f}%\n(n={int(row['count'])})", 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, stroke_pct['percentage'].max() * 1.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{plot_num:02d}_{feature_name}_vs_stroke.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_num:02d}_{feature_name}_vs_stroke.png")
    
    # Insight
    risk_ratio = stroke_pct.loc[1, 'percentage'] / stroke_pct.loc[0, 'percentage']
    print(f"  💡 Insight: {feature_name}=1 has {risk_ratio:.2f}x higher stroke rate")
    if risk_ratio > 2:
        print(f"     → STRONG medical predictor!")

analyze_binary_risk_factor('hypertension', 5)
analyze_binary_risk_factor('heart_disease', 6)

# ============================================================================
# PHASE 4: LIFESTYLE FACTORS
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: LIFESTYLE FACTORS")
print("="*80)

def analyze_categorical_feature(feature_name, plot_num):
    """Analyze categorical feature with percentage-based plot"""
    print(f"\n🏃 Analyzing {feature_name}...")
    
    # Calculate stroke percentage for each category
    stroke_pct = df.groupby(feature_name)[TARGET].agg(['sum', 'count'])
    stroke_pct['percentage'] = (stroke_pct['sum'] / stroke_pct['count']) * 100
    stroke_pct = stroke_pct.sort_values('percentage', ascending=False)
    
    print(f"  Stroke Rate by {feature_name}:")
    for idx, row in stroke_pct.iterrows():
        print(f"    {idx}: {row['percentage']:.2f}% ({int(row['sum'])}/{int(row['count'])})")
    
    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars = ax.bar(range(len(stroke_pct)), stroke_pct['percentage'], 
                  color=sns.color_palette("viridis", len(stroke_pct)), 
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel(feature_name.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.set_ylabel('Stroke Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Stroke Percentage by {feature_name.replace("_", " ").title()}\n(Sorted by Risk)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(stroke_pct)))
    ax.set_xticklabels(stroke_pct.index, rotation=45, ha='right', fontsize=10)
    
    # Add percentage annotations
    for i, (idx, row) in enumerate(stroke_pct.iterrows()):
        ax.text(i, row['percentage'] + 0.2, f"{row['percentage']:.2f}%", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylim(0, stroke_pct['percentage'].max() * 1.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{plot_num:02d}_{feature_name}_vs_stroke.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_num:02d}_{feature_name}_vs_stroke.png")
    
    # Insight
    highest_risk = stroke_pct.index[0]
    lowest_risk = stroke_pct.index[-1]
    risk_diff = stroke_pct.iloc[0]['percentage'] - stroke_pct.iloc[-1]['percentage']
    print(f"  💡 Insight: Highest risk = '{highest_risk}' ({stroke_pct.iloc[0]['percentage']:.2f}%)")
    print(f"             Lowest risk = '{lowest_risk}' ({stroke_pct.iloc[-1]['percentage']:.2f}%)")
    print(f"             Difference = {risk_diff:.2f} percentage points")

analyze_categorical_feature('smoking_status', 7)
analyze_categorical_feature('work_type', 8)
analyze_categorical_feature('Residence_type', 9)

# ============================================================================
# PHASE 5: DEMOGRAPHIC EFFECTS
# ============================================================================
print("\n" + "="*80)
print("PHASE 5: DEMOGRAPHIC EFFECTS")
print("="*80)

analyze_categorical_feature('gender', 10)
analyze_categorical_feature('ever_married', 11)

# ============================================================================
# PHASE 6: ENGINEERED FEATURES
# ============================================================================
print("\n" + "="*80)
print("PHASE 6: ENGINEERED FEATURES")
print("="*80)

analyze_categorical_feature('age_group', 12)
analyze_categorical_feature('bmi_category', 13)

# ============================================================================
# PHASE 7: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PHASE 7: CORRELATION ANALYSIS")
print("="*80)

print("\n📊 Computing correlation matrix...")

# Select numeric features for correlation
corr_features = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'stroke']
corr_matrix = df[corr_features].corr()

print("\nCorrelation with Stroke:")
stroke_corr = corr_matrix['stroke'].sort_values(ascending=False)
for feature, corr_val in stroke_corr.items():
    if feature != 'stroke':
        print(f"  {feature:20s}: {corr_val:+.4f}")

# Plot correlation heatmap
fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            mask=mask, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation Heatmap: Numeric Features\n(Lower Triangle Only)', 
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '14_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 14_correlation_heatmap.png")

print("\n💡 Insights:")
print(f"   - Strongest predictor: {stroke_corr.index[1]} (r={stroke_corr.iloc[1]:+.4f})")
print(f"   - Check for multicollinearity among predictors")

# ============================================================================
# PHASE 8: ADVANCED INSIGHTS (OPTIONAL)
# ============================================================================
print("\n" + "="*80)
print("PHASE 8: ADVANCED INSIGHTS (OPTIONAL)")
print("="*80)

print("\n🎨 Creating pairplot (this may take a moment)...")

# Sample data for faster plotting if dataset is large
sample_size = min(2000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

# Create pairplot
pairplot_features = ['age', 'avg_glucose_level', 'bmi', 'stroke']
g = sns.pairplot(df_sample[pairplot_features], hue='stroke', 
                 palette={0: '#3498db', 1: '#e74c3c'},
                 diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30},
                 height=2.5)
g.fig.suptitle('Pairplot: Numeric Features Colored by Stroke\n(Sample of 2000 records)', 
               fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '15_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 15_pairplot.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎯 FINAL SUMMARY OF KEY INSIGHTS")
print("="*80)

print("\n1. CLASS IMBALANCE:")
print(f"   - Dataset is heavily imbalanced: {stroke_pct[0]:.1f}% no stroke vs {stroke_pct[1]:.1f}% stroke")
print(f"   - Justifies focus on RECALL over accuracy in modeling")

print("\n2. STRONGEST PREDICTORS (by correlation):")
for i, (feature, corr_val) in enumerate(stroke_corr.items(), 1):
    if feature != 'stroke' and i <= 3:
        print(f"   {i}. {feature}: r={corr_val:+.4f}")

print("\n3. MEDICAL RISK FACTORS:")
print(f"   - Hypertension and heart disease significantly increase stroke probability")

print("\n4. NUMERIC FEATURES:")
print(f"   - Age shows strongest separation between stroke/no-stroke groups")
print(f"   - Glucose levels associate with higher stroke probability")
print(f"   - BMI shows weak to moderate effect")

print("\n5. LIFESTYLE FACTORS:")
print(f"   - Smoking status shows moderate association")
print(f"   - Work type may be age-confounded")
print(f"   - Residence type shows minimal impact")

print("\n" + "="*80)
print(f"✅ EDA COMPLETE! All 15 plots saved to: {OUTPUT_DIR.absolute()}")
print("="*80)
