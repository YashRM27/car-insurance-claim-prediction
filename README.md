# 🚗 Car Insurance Claim Prediction – Single Best Feature Model

## 📌 Project Overview

Insurance companies aim to accurately predict whether a customer will make an insurance claim in order to optimize pricing and reduce risk.  

In this project, we analyze customer data from **On the Road Car Insurance** to identify **the single feature that best predicts whether a customer will file a claim**, measured by **model accuracy**.

The goal is to support a **simple, interpretable model** that can be easily deployed in production with minimal infrastructure.

---

![Car](/car.jpg)

---

## 🧠 Problem Statement

Given a dataset of customer demographics, driving history, and vehicle information, determine:

> **Which single feature results in the most accurate model for predicting insurance claims?**

- Target variable: `outcome`  
  - `0` → No claim  
  - `1` → Claim made
- Constraint:  
  - Use **only one feature at a time**
  - Exclude the `id` column

---

## 📊 Dataset Description

The dataset (`car_insurance.csv`) contains customer-level insurance data.

| Column | Description |
|------|------------|
| id | Unique customer identifier |
| age | Age group (0–3) |
| gender | 0 = Female, 1 = Male |
| driving_experience | Years of driving experience (0–3) |
| education | Education level (0–2) |
| income | Income level (0–3) |
| credit_score | Credit score (0–1) |
| vehicle_ownership | 0 = Financed, 1 = Owned |
| vehicle_year | 0 = Before 2015, 1 = 2015+ |
| married | 0 = Not married, 1 = Married |
| children | Number of children |
| postal_code | Postal code |
| annual_mileage | Miles driven per year |
| vehicle_type | 0 = Sedan, 1 = Sports car |
| speeding_violations | Number of speeding violations |
| duis | Number of DUI offenses |
| past_accidents | Number of past accidents |
| outcome | Claim made (0/1) |

---

## ⚙️ Methodology

1. **Data Cleaning**
   - Missing values in:
     - `credit_score`
     - `annual_mileage`
   - Filled using **mean imputation**

2. **Modeling Approach**
   - Built **one logistic regression model per feature**
   - Formula used:
     ```python
     outcome ~ feature
     ```
   - Used `statsmodels` for model fitting

3. **Evaluation Metric**
   - Accuracy calculated from the confusion matrix:
     \[
     Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
     \]

4. **Feature Selection**
   - Compared accuracy scores across all single-feature models
   - Selected the feature with the **highest accuracy**

---

## 🏆 Results

The best performing single-feature model is:

| Best Feature | Accuracy |
|-------------|----------|
| **`driving_experience`** | **`0.7771`** |

> This feature provides the strongest standalone signal for predicting insurance claims and is recommended as the first production model.

---

## 📦 Output

The final result is stored in a pandas DataFrame:

```python
best_feature_df
