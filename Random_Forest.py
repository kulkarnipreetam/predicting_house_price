import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings
from scipy.stats import randint

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV

warnings.simplefilter(action='ignore', category=FutureWarning)

train_df = pd.read_csv(r"./Data/train.csv")

print("\nBelow are the datatypes of the input columns:\n")

print(train_df.dtypes.to_string())

for column in train_df.columns:
    if train_df[column].dtype == float or train_df[column].dtype == int:
        train_df[column] = train_df[column].fillna(0)
        
train_df[train_df.select_dtypes(include='float').columns] = train_df.select_dtypes(include='float').round().astype(int)

categorical_columns = [col for col in train_df.columns if train_df[col].dtype == object]

integer_columns = [col for col in train_df.columns if train_df[col].dtype != object]

integer_columns.remove("Id")

integer_columns_without_target = [col for col in integer_columns if col != "SalePrice"]

# Onehotencoding for categorical columns
train_df_categorical = train_df[categorical_columns].copy()

train_df_int = train_df[integer_columns_without_target].copy()

onehot_encoder = OneHotEncoder(sparse_output=False,drop='first',handle_unknown='ignore')

onehot_encoded_data = onehot_encoder.fit_transform(train_df_categorical)

onehot_encoded_df = pd.DataFrame(onehot_encoded_data, columns=onehot_encoder.get_feature_names_out(categorical_columns))

train_df_onehot_encoded = pd.concat([train_df_int, onehot_encoded_df], axis=1)

#Training a preliminary xgb model

X = train_df_onehot_encoded.copy()

y = np.log1p(train_df["SalePrice"].copy())

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators = 1000,
                                     max_depth = 15,
                                     min_samples_split = 3,
                                     min_samples_leaf = 3,
                                     max_features= 'sqrt',
                                     criterion = 'squared_error',
                                     bootstrap = True,
                                     random_state = 42)

rf_regressor.fit(X_train, y_train)

train_R2 = rf_regressor.score(X_train,y_train)

val_R2 = rf_regressor.score(X_val,y_val)

print(f"\nInitial model training R_square: {train_R2}")

print(f"\nInitial model validation R_square: {val_R2}")

feature_importance = rf_regressor.feature_importances_

feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature importance:")
print(importance_df.to_string())

'''
Identifying ideal threshold/s to drop features below an importance value and retraining the model
'''

Feature_selection_df = pd.DataFrame(columns = ["Threshold", "No_of_features_dropped", "R2"])

Feature_selection_df = Feature_selection_df.astype({"Threshold": "float", "No_of_features_dropped": "int", "R2": "float"})

threshold_values = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 2 * 10**-2, 3 * 10**-2, 4 * 10**-2, 5 * 10**-2, 6 * 10**-2]

Feature_selection_df["Threshold"] = threshold_values

for threshold in threshold_values:
    
    features_with_low_importance = [col for col in importance_df["Feature"] if importance_df.loc[importance_df["Feature"] == col, "Importance"].values[0] <= threshold]
    
    Feature_selection_df.loc[Feature_selection_df["Threshold"]==threshold,"No_of_features_dropped"] = len(features_with_low_importance)

    # Fitting a model after dropping features
    
    X_modified = X_temp.copy()
    
    y_modified = y_temp.copy()
    
    X_modified.drop(columns = features_with_low_importance, inplace = True)
    
    X_modified_train, X_modified_val, y_modified_train, y_modified_val = train_test_split(X_modified, y_modified, test_size=0.15, random_state=42)
    
    rf_regressor.fit(X_modified_train, y_modified_train)
    
    Feature_selection_df.loc[Feature_selection_df["Threshold"]==threshold,"R2"] = rf_regressor.score(X_modified_val,y_modified_val)

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Threshold of Importance')
ax1.set_ylabel('No of Features Dropped', color=color)
ax1.plot(Feature_selection_df["Threshold"], Feature_selection_df["No_of_features_dropped"], marker='o', linestyle='--', color=color, label='Number of features dropped')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Validation R2 of the RF model', color=color)
ax2.plot(Feature_selection_df["Threshold"], Feature_selection_df["R2"], marker='o', linestyle='-', color=color, label='R2')
ax2.tick_params(axis='y', labelcolor=color)

plt.xscale('log')
fig.tight_layout()
plt.title('Number of features dropped vs. Validation R2')

plt.show()

'''
Looking at the threshold vs. validation R2 graph, we can see that 0.001 is appropriate as it performs 
well without dropping too many predictors.
'''
# Hyperparameter tuning

features_dropped_from_model = [col for col in importance_df["Feature"] if importance_df.loc[importance_df["Feature"] == col, "Importance"].values[0] <= 0.002]

print("\nNumber of features dropped: {}".format(len(features_dropped_from_model)))

X_modified = X_temp.copy()

y_modified = y_temp.copy()

X_modified.drop(columns = features_dropped_from_model, inplace = True)

X_modified_train, X_modified_val, y_modified_train, y_modified_val = train_test_split(X_modified, y_modified, test_size=0.15, random_state=42)

model = RandomForestRegressor()

param_grid = {'n_estimators': randint(100, 1000),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 5),
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 'log2'],
                'criterion': ['squared_error'],
                'bootstrap': [True]}

random_search = RandomizedSearchCV(estimator=model,
                                    param_distributions=param_grid,  
                                    n_iter=200,                       
                                    cv=5,
                                    scoring='r2',
                                    verbose=1,
                                    random_state=42,
                                    n_jobs=-1
                                )

random_search.fit(X_modified_train, y_modified_train)

best_params = random_search.best_params_

print("\nBest Parameters:", best_params)

optimized_rf_model = RandomForestRegressor(**best_params)

optimized_rf_model.fit(X_modified_train, y_modified_train)

X_test_selected = X_test[X_modified.columns]

Training_R2 = optimized_rf_model.score(X_modified_train, y_modified_train)
Final_R2 = optimized_rf_model.score(X_test_selected, y_test)

print("\nBelow are the features in the final model:")
for feature in X_modified.columns:
    print("\n" + feature)

print(f"\nFinal model training accuracy (R_square): {Training_R2}")
print(f"\nFinal model test accuracy (R_square): {Final_R2}")

'''
Final submission of test results to kaggle
'''
test_df = pd.read_csv(r"./Data/test.csv")

for column in test_df.columns:
    if test_df[column].dtype == float or test_df[column].dtype == int:
        test_df[column] = test_df[column].fillna(0)
        
test_df[test_df.select_dtypes(include='float').columns] = test_df.select_dtypes(include='float').round().astype(int)

# Onehotencoding for categorical columns
test_df_categorical = test_df[categorical_columns].copy()

test_df_int = test_df[integer_columns_without_target].copy()

onehot_encoded_test_categorical_data = onehot_encoder.transform(test_df_categorical)

onehot_encoded_test_categorical_df = pd.DataFrame(onehot_encoded_test_categorical_data, columns=onehot_encoder.get_feature_names_out(categorical_columns))

onehot_encoded_test_df = pd.concat([test_df_int, onehot_encoded_test_categorical_df], axis=1)

onehot_encoded_test_df = onehot_encoded_test_df.reindex(columns=X_modified.columns, fill_value=0)

log_preds = optimized_rf_model.predict(onehot_encoded_test_df)

y_pred = np.expm1(log_preds)

submission = pd.DataFrame(columns = ['Id', 'SalePrice'])

submission['Id'] = test_df['Id']
submission['SalePrice'] = y_pred
print("\nBelow are the predictions on the unlabeled test data:")
print(submission)
submission.to_csv('submission.csv', index=False)