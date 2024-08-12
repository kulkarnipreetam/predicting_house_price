import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)

train_df = pd.read_csv(r"/kaggle/input/house-prices-advanced-regression-techniques/train.csv") #Use appropriate filepath here

#Handle missing values
for column in train_df.columns:
    if train_df[column].dtype == object:
        train_df[column] = train_df[column].fillna("Not_Available")

for column in train_df.columns:
    if train_df[column].dtype == float or train_df[column].dtype == int:
        train_df[column] = train_df[column].fillna(0)

categorical_columns = [col for col in train_df.columns if train_df[col].dtype == object]

integer_columns = [col for col in train_df.columns if train_df[col].dtype != object]

integer_columns.remove("Id")

integer_columns_without_target = [col for col in integer_columns if col != "SalePrice"]

#Training a catboost model

X = train_df.drop(columns=["SalePrice", "Id"]).copy()

y = train_df["SalePrice"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

catboost_regressor = CatBoostRegressor(iterations = 500,
                                       learning_rate = 0.1,
                                       cat_features = categorical_columns,
                                       loss_function = "RMSE",
                                       silent = True)

catboost_regressor.fit(X_train, y_train)

R2 = catboost_regressor.score(X_test,y_test)

print(f"\nInitial model accuracy (R_square): {R2}")

# Feature importance

feature_importance = catboost_regressor.feature_importances_

feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature importance:")
print(importance_df.to_string())

'''
Identifying ideal threshold/s to drop features below an inportance value and retraining the model
'''

Feature_selection_df = pd.DataFrame(columns = ["Threshold", "No_of_features_dropped", "R2"])

Feature_selection_df = Feature_selection_df.astype({"Threshold": "float", "No_of_features_dropped": "int", "R2": "float"})

threshold_values = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

Feature_selection_df["Threshold"] = threshold_values

for threshold in threshold_values:
    
    features_with_low_importance = [col for col in importance_df["Feature"] if importance_df.loc[importance_df["Feature"] == col, "Importance"].values[0] <= threshold]
    
    Feature_selection_df.loc[Feature_selection_df["Threshold"]==threshold,"No_of_features_dropped"] = len(features_with_low_importance)

    # Fitting a model after dropping features
    
    X_modified = X.copy()
    
    y = train_df["SalePrice"].copy()
    
    X_modified.drop(columns = features_with_low_importance, inplace = True)
    
    X_modified_train, X_modified_test, y_train, y_test = train_test_split(X_modified, y, test_size=0.2, random_state=42)
    
    categorical_columns_modified = [col for col in X_modified.columns if X_modified[col].dtype == object]
    
    catboost_regressor_modified = CatBoostRegressor(iterations = 200,
                                                    learning_rate = 0.1,
                                                    cat_features = categorical_columns_modified,
                                                    loss_function = "RMSE",
                                                    silent = True)
    
    catboost_regressor_modified.fit(X_modified_train, y_train)
    
    Feature_selection_df.loc[Feature_selection_df["Threshold"]==threshold,"R2"] = catboost_regressor_modified.score(X_modified_test,y_test)

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Threshold of Importance')
ax1.set_ylabel('No of Features Dropped', color=color)
ax1.plot(Feature_selection_df["Threshold"], Feature_selection_df["No_of_features_dropped"], marker='o', linestyle='--', color=color, label='Number of features dropped')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('R2 of the RF model', color=color)
ax2.plot(Feature_selection_df["Threshold"], Feature_selection_df["R2"], marker='o', linestyle='-', color=color, label='R2')
ax2.tick_params(axis='y', labelcolor=color)

plt.xscale('log')
fig.tight_layout()
plt.title('Number of features dropped vs. R2')

plt.show()

'''
Looking at the threshold vs. R2 graph, 1.5, 2.5 and 3.5 can be a good choice of threshold as the R2 is stable upto 3.5 
without dropping too many features. Need to evaluate if better accuracy can be achieved through hyperparameter tuning while 
retaining more features i.e., by using thresholds of 1.5 and 2.5.
'''
thresholds_for_further_evaluation = [1.5, 2.5, 3.5]
hyperparameter_dict = {}

print("\nEvaluating thresholds to drop features with hyperparameter tuning:")
for i in range(len(thresholds_for_further_evaluation)):

    # Hyperparameter tuning
    
    threshold_being_evaluated = thresholds_for_further_evaluation[i]
    
    features_dropped_from_model = [col for col in importance_df["Feature"] if importance_df.loc[importance_df["Feature"] == col, "Importance"].values[0] <= threshold_being_evaluated]
    
    print("\nThreshold {} : {}".format(i+1,thresholds_for_further_evaluation[i]))
    print("\nNumber of features dropped: {}".format(len(features_dropped_from_model)))
    
    X_modified = X.copy()
    
    y = train_df["SalePrice"].copy()
    
    X_modified.drop(columns = features_dropped_from_model, inplace = True)
    
    X_modified_train, X_modified_test, y_train, y_test = train_test_split(X_modified, y, test_size=0.2, random_state=42)
    
    categorical_columns_modified = [col for col in X_modified.columns if X_modified[col].dtype == object]
    
    model = CatBoostRegressor(cat_features = categorical_columns_modified, silent = True)
    
    param_grid = {'iterations': [100, 200, 400],
                  'max_depth': [4, 5, 6, 7],
                  'learning_rate': [0.01, 0.05, 0.1],
                  'loss_function': ['RMSE', 'MAE']}
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=0)
    
    grid_search.fit(X_modified_train, y_train)
    
    best_params = grid_search.best_params_
    
    hyperparameter_dict[thresholds_for_further_evaluation[i]] = best_params
    
    print("\nBest Parameters:", best_params)
    
    optimized_catboost_model = CatBoostRegressor(**best_params, cat_features = categorical_columns_modified, silent = True)
    
    optimized_catboost_model.fit(X_modified_train, y_train)
    
    R2 = optimized_catboost_model.score(X_modified_test,y_test)
    
    print("\nR2 of model after hyperparameter tuning: {}".format(R2))
    
'''
After evaluating thresholds while varying hyperparameters, we can see that 1.5 is appropriate as it performs 
better when compared with other thresholds i.e., a model with 13 features is better. 
'''

ideal_threshold = 1.5

print("\nIdeal threshold to drop features: {}".format(ideal_threshold))

features_dropped_from_final_model = [col for col in importance_df["Feature"] if importance_df.loc[importance_df["Feature"] == col, "Importance"].values[0] <= ideal_threshold]

X_final = X.copy()

y = train_df["SalePrice"].copy()

X_final.drop(columns = features_dropped_from_final_model, inplace = True)

X_final_train, X_final_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

categorical_columns_final = [col for col in X_final.columns if X_final[col].dtype == object]

final_model = CatBoostRegressor(**hyperparameter_dict[ideal_threshold], cat_features = categorical_columns_final, silent = True)

final_model.fit(X_final_train, y_train)

R2 = final_model.score(X_final_test,y_test)

print("\nBelow are the features in the final model:")
for feature in X_final.columns:
    print(feature)

print(f"\nFinal model accuracy (R_square): {R2}")
