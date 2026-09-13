# CodeAlpha_ExploratoryDataAnalysis

## Task
Exploratory Data Analysis (EDA) on the Titanic dataset using Python (Pandas, Matplotlib, Seaborn, SciPy).

## Objective
To explore a real-world dataset systematically — asking meaningful questions first, understanding data structure and types, finding trends/patterns, testing a hypothesis statistically, and detecting data quality issues before any further analysis or modeling.

## Tools & Libraries Used
- Python 3
- `pandas` – data loading, cleaning, and exploration
- `numpy` – numerical operations
- `matplotlib` – base plotting engine
- `seaborn` – statistical visualizations (count plots, heatmaps, boxplots)
- `scipy` – chi-square hypothesis testing

## Dataset Used
**Titanic dataset** — a classic, free, public dataset containing passenger details (age, sex, class, fare, survival status) from the Titanic. It is small, well documented, and has real-world data issues (missing values, outliers), making it ideal for practicing EDA.

- Primary source (used automatically if internet is available):
  `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv`
- `titanic_sample.csv` is included in this repo as an **offline fallback** with the same column structure, so the script always runs even without internet access.

## Folder Structure
```
CodeAlpha_ExploratoryDataAnalysis/
│
├── eda_titanic.py          # Main EDA script
├── titanic_sample.csv      # Offline fallback sample dataset
├── README.md                # Project description
└── requirements.txt         # Required libraries
```
(Chart images `chart1_...png` to `chart6_...png` are generated automatically when you run the script.)

## Questions Asked Before Analysis
1. What percentage of passengers survived?
2. Did gender affect survival chances?
3. Did passenger class (1st/2nd/3rd) affect survival?
4. Does age affect survival chances?
5. Are there missing values or data quality issues?
6. Is there a relationship between fare and class/survival?

## How to Run
1. Clone this repository:
   ```
   git clone https://github.com/ramprasadshinde9834-star/CodeAlpha_ExploratoryDataAnalysis.git
   cd CodeAlpha_ExploratoryDataAnalysis
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the script:
   ```
   python eda_titanic.py
   ```
4. Output:
   - Console prints: dataset shape, data types, missing values, survival rate, and chi-square test result
   - 6 chart images (`.png`) are saved in the same folder

## Sample Output (Console)
```
Dataset loaded successfully from online source.

--- Shape of dataset (rows, columns) ---
(891, 15)

--- Missing values per column ---
age            177
embarked         2
deck           688

Overall survival rate: 38.38%

Chi-square test (Gender vs Survival): p-value = 0.00000
Result: Gender has a statistically significant effect on survival.

EDA completed! 6 charts saved as PNG files in this folder.
```

## Charts Generated
| File | What it shows |
|---|---|
| chart1_survival_count.png | Overall survivors vs non-survivors |
| chart2_survival_by_gender.png | Survival split by gender |
| chart3_survival_by_class.png | Survival split by passenger class |
| chart4_age_distribution.png | Age distribution of survivors vs non-survivors |
| chart5_fare_by_class.png | Fare spread and outliers per class |
| chart6_correlation_heatmap.png | Correlation between numeric variables |

## Key Findings
- Missing values found in `age`, `embarked`, and heavily in `deck` — needs handling before modeling.
- Overall survival rate is around 33–38%.
- Gender has a statistically significant effect on survival (confirmed via chi-square test, p-value < 0.05) — women had a much higher survival rate than men.
- 1st class passengers had a noticeably higher survival rate than 3rd class.
- Fare values contain outliers — a few passengers paid far higher fares than others in the same class.

## Real-World Applications
- Healthcare: exploring patient records to spot risk factors before building prediction models
- Banking/Finance: exploring transaction data to detect anomalies or fraud
- Retail: understanding customer purchase patterns before building recommendation systems
- HR Analytics: exploring employee data to find attrition patterns

## Author
Ramprasad Shinde

## Internship
This project was completed as part of the **CodeAlpha Internship Program**.
