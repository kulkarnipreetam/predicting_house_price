# 🏠 House Price Prediction – Kaggle Competition

This repository contains my solution to the [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition. The objective is to predict the final sale prices of homes in Ames, Iowa, using various regression techniques.

### 1. **Data Exploration & Cleaning**
- Imputed missing values:
  - Numerical features: filled with 0
  - Categorical features: filled with `"Missing"`
- Visualized feature correlations and relationships

### 2. **Feature Engineering**
- One-hot encoding for models like Random Forest and XGBoost
- Used CatBoost's native categorical handling (via `cat_features`)
- Feature selection based on importance thresholding

### 3. **Modeling & Evaluation**
Models used:
- ✅ **Random Forest**
- ✅ **XGBoost**
- ✅ **CatBoost**

All models were tuned using `RandomizedSearchCV` with cross-validation (`cv=5`, `n_iter=50`) and evaluated using:
- **Root Mean Squared Log Error (RMSLE)** (per Kaggle's evaluation)
> Target variable (`SalePrice`) was log-transformed (`log1p`) to align with evaluation metric.

| Model         |  Training R² |  Test R² (Holdout) |
|---------------|--------------|--------------------|
| Random Forest | ~0.955       | ~0.857             |
| XGBoost       | ~0.942       | ~0.896             |
| CatBoost      | ~0.953       | ~0.920             |

---

## 🧠 Key Insights

- **CatBoost outperforms** both Random Forest and XGBoost on the test set, achieving the highest test R² (~0.92) with strong generalization from training.
- **XGBoost** also performs very well, with good balance between training (~0.94) and test (~0.90) R², showing effective handling of feature interactions.
- **Random Forest** achieves the highest training R² (~0.96) but shows relatively lower test R² (~0.86), indicating slight overfitting compared to gradient boosting models.
- Log-transforming the target variable (`SalePrice`) improves model stability and aligns well with the competition's RMSLE metric.
- CatBoost's native categorical feature handling simplifies preprocessing and contributes to its superior performance.
- Careful hyperparameter tuning and feature selection significantly improve model accuracy and generalization.

---

## 🚀 Future Improvements

- Try ensembling (e.g., blending or stacking)
- Add location-based engineered features
- Test LightGBM and other gradient-boosting alternatives
- Implement SHAP or permutation importance for interpretability

---

## Important Notice

The code in this repository is proprietary and protected by copyright law. Unauthorized copying, distribution, or use of this code is strictly prohibited. By accessing this repository, you agree to the following terms:

- **Do Not Copy:** You are not permitted to copy any part of this code for any purpose.
- **Do Not Distribute:** You are not permitted to distribute this code, in whole or in part, to any third party.
- **Do Not Use:** You are not permitted to use this code, in whole or in part, for any purpose without explicit permission from the owner.

If you have any questions or require permission, please contact the repository owner.

Thank you for your cooperation.
