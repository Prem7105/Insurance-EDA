# Medical Insurance Cost Prediction — EDA & Feature Engineering

Exploratory data analysis, cleaning, feature engineering, and feature selection on a medical insurance dataset, in preparation for a regression model predicting individual medical charges.

## Dataset

`insurance.csv` — 1,338 records, 6 features + 1 target.

| Column | Description |
|---|---|
| age | Age of the primary beneficiary |
| sex | male / female |
| bmi | Body mass index |
| children | Number of dependents covered |
| smoker | yes / no |
| region | northeast, northwest, southeast, southwest |
| **charges** | Target: individual medical costs billed by insurance |

## What's in the notebook (`insurance.ipynb`)

1. **EDA** — shape, dtypes, summary stats, null checks, distribution histograms (`age`, `bmi`, `children`, `charges`), count plots (`sex`, `smoker`, `children`), box plots for outliers, correlation heatmap.
2. **Cleaning** — drop duplicate rows; encode `sex` → `is_female` and `smoker` → `is_smoker` (0/1); one-hot encode `region`.
3. **Feature engineering** — derived `bmi_category` (Underweight / Normal / Overweight / Obese), one-hot encoded, then `StandardScaler` applied to `age`, `bmi`, `children`.
4. **Feature selection** — Pearson correlation for numeric features against `charges`; Chi-square test of independence for categorical features against a quartile-binned version of `charges`. Final feature set: `age`, `is_female`, `bmi`, `children`, `is_smoker`, `region_southeast`, `bmi_category_Obese`.

## Status

EDA, cleaning, feature engineering, and feature selection are complete. Model training/evaluation is not yet implemented — planned as a next step.

## Getting started

```bash
pip install numpy pandas seaborn matplotlib scikit-learn scipy
jupyter notebook insurance.ipynb
```
