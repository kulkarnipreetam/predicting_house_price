# Predicting house sale price 
The goal of this kaggle project/competition is to develop a machine learning model to predict the sale price of a house based on a training dataset with 80 features and 1460 instances.

### Exploratory data analysis 
- Identified features that are strongly correlated with the target variable (sale price)
- Detected highly correlated features that might not contribute additional information
- Performed Principal Component Analysis (PCA) to assess whether the variation in the features is largely explained by the first or second principal components

### Modeling
Random Forest, CatBoost, and XGBoost models were trained on the dataset. Feature engineering and hyperparameter tuning were performed together to optimize model performance. Various thresholds were identified based on the range of feature importance metrics. Features with importance below these thresholds were dropped, followed by further hyperparameter tuning. The best combination of threshold and hyperparameters was selected.

### Result
Xgboost emerged as the best performing model with an R2 of 91.6%.

## Important Notice

The code in this repository is proprietary and protected by copyright law. Unauthorized copying, distribution, or use of this code is strictly prohibited. By accessing this repository, you agree to the following terms:

- **Do Not Copy:** You are not permitted to copy any part of this code for any purpose.
- **Do Not Distribute:** You are not permitted to distribute this code, in whole or in part, to any third party.
- **Do Not Use:** You are not permitted to use this code, in whole or in part, for any purpose without explicit permission from the owner.

If you have any questions or require permission, please contact the repository owner.

Thank you for your cooperation.
