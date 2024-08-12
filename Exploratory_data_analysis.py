import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

warnings.simplefilter(action='ignore', category=FutureWarning)

train_df = pd.read_csv(r"/kaggle/input/house-prices-advanced-regression-techniques/train.csv") #Use appropriate filepath here

# Data types
print("\nSummary of data types:")
print(train_df.dtypes.to_string())

#Handle missing values
print("\nMissing Values in categorical features before imputing:\n")
for col in train_df.columns:
    if train_df[col].isna().sum() != 0 and train_df[col].dtype == object:
        print("{} has {} missing values".format(col,train_df[col].isna().sum()))

for column in train_df.columns:
    if train_df[column].dtype == object:
        train_df[column] = train_df[column].fillna("Not_Available")

print("\nMissing Values in numerical features before imputing:\n")
for col in train_df.columns:
    if train_df[col].isna().sum() != 0 and train_df[col].dtype != object:
        print("{} has {} missing values".format(col,train_df[col].isna().sum()))

for column in train_df.columns:
    if train_df[column].dtype == float or train_df[column].dtype == int:
        train_df[column] = train_df[column].fillna(0)

categorical_columns = [col for col in train_df.columns if train_df[col].dtype == object]

integer_columns = [col for col in train_df.columns if train_df[col].dtype != object]

integer_columns.remove("Id")

integer_columns_without_target = [col for col in integer_columns if col != "SalePrice"]

# Finding feature pairs with high correlation
train_df_coded = train_df.copy()

for i in categorical_columns:
    train_df_coded[i + '_coded'] = LabelEncoder().fit_transform(train_df_coded[i])

train_df_coded = train_df_coded.drop(categorical_columns, axis = 1)

train_df_coded_corr_matrix = train_df_coded.corr()

high_correlation_pairs = (train_df_coded_corr_matrix.abs() >= 0.6) & (train_df_coded_corr_matrix != 1)

upper_triangle_indices = zip(*np.triu_indices_from(train_df_coded_corr_matrix, k=1))

unique_pairs = [(train_df_coded_corr_matrix.index[i], 
                 train_df_coded_corr_matrix.columns[j], 
                 train_df_coded_corr_matrix.iloc[i, j]) for i, j in upper_triangle_indices if high_correlation_pairs.iloc[i, j]]

unique_pairs_df = pd.DataFrame(unique_pairs, columns=['Feature_1', 'Feature_2', 'Correlation'])

print("\nUnique Pairs with High Correlation:")
print(unique_pairs_df)

'''
It is observed from the correlation matrix that the sales price of a house is strongly correlated with the following features:
    1) OverallQual - Overall material and finish quality
    2) TotalBsmtSF - Total square feet of basement area
    3) 1stFlrSF - First Floor square feet
    4) GrLivArea - Above grade (ground) living area square feet
    5) GarageCars - Size of garage in car capacity
    6) GarageArea - Size of garage in square feet
    7) ExterQual - Exterior material quality
'''

# individual plots i.e., sales price vs feature
feature_list = [col for col in train_df_coded.columns if col not in ["SalePrice", "Id"]]

for col in feature_list:
    g = sns.jointplot(data=train_df_coded, x=col, y="SalePrice")
    g.fig.suptitle(f"SalePrice versus {col}", y=1.03)
    plt.show()


#Vizualizing and understanding the target

sns.histplot(data=train_df, x="SalePrice", kde=True)
plt.title("Distribution of house sale price")

print("\nSummary statistics of Sales price:")
print(train_df["SalePrice"].describe())

# Identifying outliers in each feature

data_for_outliers = train_df[integer_columns].copy()

outliers_index_df = pd.DataFrame()

for i in data_for_outliers.columns:
    Q1 = data_for_outliers[i].quantile(0.25)
    Q3 = data_for_outliers[i].quantile(0.75)
    IQR = Q3 - Q1
    threshold = 1.5
    
    x = train_df[(data_for_outliers[i] < Q1 - threshold * IQR) | (data_for_outliers[i] > Q3 + threshold * IQR)]
    outliers_index_df [i] = pd.Series(x.index)

print("\nThe number of outliers identified in each column:\n")
print(outliers_index_df.count())

# Onehotencoding for categorical columns and scalarizing integer columns
train_df_categorical = train_df[categorical_columns].copy()

train_df_int = train_df[integer_columns_without_target].copy()

scaler = StandardScaler()

onehot_encoder = OneHotEncoder(sparse=False, drop='first')

scaled_data = scaler.fit_transform(train_df_int)

onehot_encoded_data = onehot_encoder.fit_transform(train_df_categorical)

train_df_scalarized = pd.DataFrame(scaled_data, columns=scaler.get_feature_names_out(integer_columns_without_target))

train_df_onehot_encoded = pd.DataFrame(onehot_encoded_data, columns=onehot_encoder.get_feature_names_out(categorical_columns))

train_df_onehot_encoded_scalarized = pd.concat([train_df_scalarized, train_df_onehot_encoded], axis=1)

# Principal component analysis (PCA)
pca = PCA()

pca.fit_transform(train_df_onehot_encoded_scalarized)

explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Eigenvalue Variance Ratio', color=color)
ax1.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--', color=color, label='Explained Variance Ratio')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Cumulative Explained Variance Ratio', color=color)
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color=color, label='Cumulative Explained Variance Ratio')
ax2.tick_params(axis='y', labelcolor=color)

ax1.grid(True)
fig.tight_layout()
plt.title('Scree Plot with Explained Variance Ratio and Cumulative Explained Variance')

plt.show()

'''
The highest eigen value ratio is 15% suggesting that there are no strong underlying patters in the input feature.
Feature data is spread out and possibly complex.
'''
