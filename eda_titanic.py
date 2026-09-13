import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

sns.set_style("whitegrid")

# -----------------------------------------------------
# STEP 1: Load the dataset
# (Tries the real online Titanic dataset first.
#  If there is no internet connection, it falls back
#  to the local sample file so the script never breaks.)
# -----------------------------------------------------
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
try:
    df = pd.read_csv(url)
    print("Dataset loaded successfully from online source.")
except Exception:
    print("Could not reach internet, using local sample file instead.")
    df = pd.read_csv("titanic_sample.csv")

# -----------------------------------------------------
# STEP 2: Explore data structure (variables & data types)
# -----------------------------------------------------
print("\n--- Shape of dataset (rows, columns) ---")
print(df.shape)

print("\n--- Column names and data types ---")
print(df.dtypes)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Statistical summary of numeric columns ---")
print(df.describe())

# -----------------------------------------------------
# STEP 3: Detect data quality issues
# -----------------------------------------------------
print("\n--- Missing values per column ---")
print(df.isnull().sum())

print("\n--- Duplicate rows ---")
print(f"Number of duplicate rows: {df.duplicated().sum()}")

# -----------------------------------------------------
# STEP 4: Question 1 - Overall survival rate
# -----------------------------------------------------
survival_rate = df['survived'].mean() * 100
print(f"\nOverall survival rate: {survival_rate:.2f}%")

plt.figure(figsize=(5, 4))
sns.countplot(x='survived', data=df)
plt.title("Survival Count (0 = Died, 1 = Survived)")
plt.savefig("chart1_survival_count.png")
plt.close()

# -----------------------------------------------------
# STEP 5: Question 2 - Does gender affect survival?
# -----------------------------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival by Gender")
plt.savefig("chart2_survival_by_gender.png")
plt.close()

contingency = pd.crosstab(df['sex'], df['survived'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test (Gender vs Survival): p-value = {p_value:.5f}")
if p_value < 0.05:
    print("Result: Gender has a statistically significant effect on survival.")
else:
    print("Result: No significant relationship found.")

# -----------------------------------------------------
# STEP 6: Question 3 - Does passenger class affect survival?
# -----------------------------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x='pclass', hue='survived', data=df)
plt.title("Survival by Passenger Class")
plt.savefig("chart3_survival_by_class.png")
plt.close()

# -----------------------------------------------------
# STEP 7: Question 4 - Does age affect survival?
# -----------------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='age', hue='survived', bins=20, kde=True)
plt.title("Age Distribution by Survival")
plt.savefig("chart4_age_distribution.png")
plt.close()

# -----------------------------------------------------
# STEP 8: Question 5 - Fare vs Class relationship (anomaly check)
# -----------------------------------------------------
plt.figure(figsize=(5, 4))
sns.boxplot(x='pclass', y='fare', data=df)
plt.title("Fare Distribution by Class (Outlier Check)")
plt.savefig("chart5_fare_by_class.png")
plt.close()

# -----------------------------------------------------
# STEP 9: Correlation heatmap (numeric variables only)
# -----------------------------------------------------
plt.figure(figsize=(7, 5))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("chart6_correlation_heatmap.png")
plt.close()

print("\nEDA completed! 6 charts saved as PNG files in this folder.")
