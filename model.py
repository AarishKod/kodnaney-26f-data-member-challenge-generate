"""
By: Aarish Kodnaney
FOR: Generate F26 Data Engineer Application
model.py
"""

import pandas as pd
import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor


# loading data created by etl pipeline
df = pd.read_parquet('./data/clean.parquet')
print(df.columns)

# creating X/y split of dataframe
X = df.drop(columns="annual_salary_usd")
y = df["annual_salary_usd"]

# splitting dataframe into test and train
# I'm using George Russell's F1 number for random state
GEORGE_RUSSELL_NUMBER = 63
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.2, random_state=GEORGE_RUSSELL_NUMBER)

# defining categorical and numerical variables
categories = ["DevType", "Industry", "RemoteWork", "Country"]
numerical_cols = X.drop(columns=categories).columns # I used 'X' instead of 'df' because 'df' still includes 'annual_salayr_usd'

# imputing NaN values (only for numerical columns)
imputer = SimpleImputer(strategy='median')
X_train_df_imputed = imputer.fit_transform(X_train_df[numerical_cols])
X_test_df_imputed = imputer.transform(X_test_df[numerical_cols]) # from what I understand, transform uses the median values generated from fit_transform() for imputation

# One-hot encoding the categorical variables
encoder = OneHotEncoder(sparse_output=False, min_frequency=30, handle_unknown='ignore')
X_train_df_encoded = encoder.fit_transform(X_train_df[categories])
X_test_df_encoded = encoder.transform(X_test_df[categories])

# recombining categorical part of array with numerical part of array
X_train: ndarray = np.hstack([X_train_df_imputed, X_train_df_encoded])
X_test: ndarray = np.hstack([X_test_df_imputed, X_test_df_encoded])

def report(name, actual, pred):
    print(f"{name:28} MAE ${mean_absolute_error(actual, pred):>9,.0f}   "
          f"R² {r2_score(actual, pred):.3f}")

# 1. baseline
median_pred = np.full(len(y_test_df), y_train_df.median())
report("baseline (train median)", y_test_df, median_pred)

# 2. multiple regression (ridge)

# part 1 - rescaling each column to mean 0 and sd 1. This places every column on the same scale, regardless of oroginal units
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train) # computes mean and std from training data, applies them
X_test_s = scaler.transform(X_test) # applies stored train values from fit_transform

ridge = Ridge(alpha=1.0) # Ridge penalizes higher coefficients. This is something wise to do when predicting something with a long tail of outliers
# alpha=1.0 means mild penalty for high coefficients
ridge.fit(X_train_s, np.log(y_train_df)) # fitting to the actual ridge model
# I used log because without it, the model would treaat predicting 500k for 520k and 20k for 40k as the same mistake
# In reality, in the 20k case, we're off by 50%, and in the 500k case, we're off by 4%. Withougt logs, the model treats both of 
# those as the same error. With the logs, we care about percentages instead of dollars, which is why I use log
prediction = np.exp(ridge.predict(X_test_s)) # utilizing e^x to convert back into salaries

report("ridge", y_test_df, prediction)

# Gradient Boosting random forest trees
gb = HistGradientBoostingRegressor(random_state=GEORGE_RUSSELL_NUMBER)
gb.fit(X_train, np.log(y_train_df))
report("gradient boosting (log target)", y_test_df, np.exp(gb.predict(X_test)))