import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

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

# Onehotencoding for categorical columns
train_df_categorical = train_df[categorical_columns].copy()

train_df_int = train_df[integer_columns_without_target].copy()

onehot_encoder = OneHotEncoder(sparse=False, drop='first')

onehot_encoded_data = onehot_encoder.fit_transform(train_df_categorical)

onehot_encoded_df = pd.DataFrame(onehot_encoded_data, columns=onehot_encoder.get_feature_names_out(categorical_columns))

train_df_onehot_encoded = pd.concat([train_df_int, onehot_encoded_df], axis=1)

#Training a xgb model

X = train_df_onehot_encoded.copy()

y = train_df["SalePrice"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_regressor = xgb.XGBRegressor(n_estimators=500,
                                 learning_rate=0.1,
                                 objective='reg:squarederror')

xgb_regressor.fit(X_train, y_train)

R2 = xgb_regressor.score(X_test,y_test)

print(f"\nInitial model accuracy (R_square): {R2}")

# Feature importance

feature_importance = xgb_regressor.feature_importances_

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

threshold_values = [10**-5, 10**-4, 10**-3, 2 * 10**-3, 4 * 10**-3, 9 * 10**-3, 10**-2, 2 * 10**-2]

Feature_selection_df["Threshold"] = threshold_values

for threshold in threshold_values:
    
    features_with_low_importance = [col for col in importance_df["Feature"] if importance_df.loc[importance_df["Feature"] == col, "Importance"].values[0] <= threshold]
    
    Feature_selection_df.loc[Feature_selection_df["Threshold"]==threshold,"No_of_features_dropped"] = len(features_with_low_importance)

    # Fitting a model after dropping features
    
    X_modified = train_df_onehot_encoded.copy()
    
    y = train_df["SalePrice"].copy()
    
    X_modified.drop(columns = features_with_low_importance, inplace = True)
    
    X_modified_train, X_modified_test, y_train, y_test = train_test_split(X_modified, y, test_size=0.2, random_state=42)
    
    xgb_regressor.fit(X_modified_train, y_train)
    
    Feature_selection_df.loc[Feature_selection_df["Threshold"]==threshold,"R2"] = xgb_regressor.score(X_modified_test,y_test)

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
Looking at the threshold vs. R2 graph, 0.001, 0.002, 0.003 and 0.004 can be a good choice of threshold as the R2 is stable upto 0.004 
without dropping too many features. Need to evaluate if better accuracy can be achieved through hyperparameter tuning while 
retaining more features i.e., by using thresholds of 0.002 or 0.001.
'''
thresholds_for_further_evaluation = [0.001, 0.002, 0.003, 0.004]
hyperparameter_dict = {}

print("\nEvaluating thresholds to drop features with hyperparameter tuning:")
for i in range(len(thresholds_for_further_evaluation)):

    # Hyperparameter tuning
    
    threshold_being_evaluated = thresholds_for_further_evaluation[i]
    
    features_dropped_from_model = [col for col in importance_df["Feature"] if importance_df.loc[importance_df["Feature"] == col, "Importance"].values[0] <= threshold_being_evaluated]
    
    print("\nThreshold {} : {}".format(i+1,thresholds_for_further_evaluation[i]))
    print("\nNumber of features dropped: {}".format(len(features_dropped_from_model)))

    X_modified = train_df_onehot_encoded.copy()
    
    y = train_df["SalePrice"].copy()
    
    X_modified.drop(columns = features_dropped_from_model, inplace = True)
    
    X_modified_train, X_modified_test, y_train, y_test = train_test_split(X_modified, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor()
    
    param_grid = {'n_estimators': [100,300, 400],
                  'max_depth': [4, 5, 6, 7],
                  'learning_rate': [0.01, 0.05, 0.1],
                  'objective': ['reg:squarederror', 'reg:pseudohubererror', 'reg:absoluteerror']}
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=0)
    
    grid_search.fit(X_modified_train, y_train)
    
    best_params = grid_search.best_params_
    
    hyperparameter_dict[thresholds_for_further_evaluation[i]] = best_params
    
    print("\nBest Parameters:", best_params)
    
    optimized_xgb_model = xgb.XGBRegressor(**best_params)
    
    optimized_xgb_model.fit(X_modified_train, y_train)
    
    R2 = optimized_xgb_model.score(X_modified_test,y_test)
    
    print("\nR2 of model after hyperparameter tuning: {}".format(R2))
    
'''
After evaluating thresholds while varying hyperparameters, we can see that 0.003 is appropriate as it performs 
the best when compared with other thresholds. 
'''

ideal_threshold = 0.003

print("\nIdeal threshold to drop features: {}".format(ideal_threshold))

features_dropped_from_final_model = [col for col in importance_df["Feature"] if importance_df.loc[importance_df["Feature"] == col, "Importance"].values[0] <= ideal_threshold]

X_final = train_df_onehot_encoded.copy()

y = train_df["SalePrice"].copy()

X_final.drop(columns = features_dropped_from_final_model, inplace = True)

X_final_train, X_final_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

final_model = xgb.XGBRegressor(**hyperparameter_dict[ideal_threshold])

final_model.fit(X_final_train, y_train)

R2 = final_model.score(X_final_test,y_test)

print("\nBelow are the features in the final model:")
for feature in X_final.columns:
    print(feature)

print(f"\nFinal model accuracy (R_square): {R2}")

'''
Xgboost gives the best accuracy compared to catboost and random forest and hence is choosen for submission of test results to kaggle
'''
test_df = pd.read_csv(r"/kaggle/input/house-prices-advanced-regression-techniques/test.csv") #Use appropriate filepath here

for column in test_df.columns:
    if test_df[column].dtype == object:
        test_df[column] = test_df[column].fillna("Not_Available")

for column in test_df.columns:
    if test_df[column].dtype == float or test_df[column].dtype == int:
        test_df[column] = test_df[column].fillna(0)

# Onehotencoding for categorical columns
test_df_categorical = test_df[categorical_columns].copy()

test_df_int = test_df[integer_columns_without_target].copy()

onehot_encoder = OneHotEncoder(sparse=False, drop='first')

onehot_encoded_test_categorical_data = onehot_encoder.fit_transform(test_df_categorical)

onehot_encoded_test_categorical_df = pd.DataFrame(onehot_encoded_test_categorical_data, columns=onehot_encoder.get_feature_names_out(categorical_columns))

onehot_encoded_test_df = pd.concat([test_df_int, onehot_encoded_test_categorical_df], axis=1)

onehot_encoded_test_df = onehot_encoded_test_df[X_final.columns]

y_pred = final_model.predict(onehot_encoded_test_df)

submission = pd.DataFrame(columns = ['Id', 'SalePrice'])

submission['Id'] = test_df['Id']
submission['SalePrice'] = y_pred
print("\nBelow are the predictions on the unlabeled test data:")
print(submission)
submission.to_csv('submission.csv', index=False)
