"""
By: Aarish Kodnaney
FOR: Generate F26 Data Engineer Application
model.py
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

df = pd.read_parquet('./data/clean.parquet')

# Numeric: already quantified in the ETL step (ordinal maps, midpoints, counts).
# ICorPM is 0/1 with NaN for self-employed respondents, who were never asked.
numeric = [
    'Age', 'EdLevel', 'Employment', 'WorkExp', 'YearsCode',
    'orgsize_log', 'LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith',
    'ICorPM', 'income_tier',
]

# Categorical: left as raw strings so the encoder can learn levels per fold.
categorical = ['DevType', 'RemoteWork', 'Industry', 'Country']

features = numeric + categorical


def make_prep():
    """Fresh preprocessor per model so no fitted state is shared."""
    return ColumnTransformer([
        ('num', make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler()
        ), numeric),
        ('cat', make_pipeline(
            SimpleImputer(strategy='constant', fill_value='Unknown'),
            OneHotEncoder(handle_unknown='ignore', min_frequency=30, sparse_output=False)
        ), categorical),
    ])


X = df[features]
y = df['annual_salary_usd']

# Stratify on salary quartile: the target is right-skewed, so a plain random
# split can load one side with high earners.
strata = pd.qcut(y, q=4, labels=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strata
)

print(f"train {len(X_train)}  test {len(X_test)}")
print(f"median  train ${y_train.median():,.0f}  test ${y_test.median():,.0f}\n")

# Train on log salary: errors are multiplicative, not additive, and the
# distribution is heavily right-skewed.
y_train_log = np.log(y_train)

models = {
    'ridge': Pipeline([
        ('prep', make_prep()),
        ('est', Ridge(alpha=1.0)),
    ]),
    'gbm': Pipeline([
        ('prep', make_prep()),
        ('est', HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    ]),
}

# --- Baselines -------------------------------------------------------------
# Anything the models produce has to beat these to be worth having.

median_pred = np.full(len(y_test), y_train.median())
print(f"{'baseline (global median)':<26} MAE ${mean_absolute_error(y_test, median_pred):>9,.0f}")

country_median = y_train.groupby(X_train.Country).median()
country_pred = X_test.Country.map(country_median).fillna(y_train.median())
print(f"{'baseline (country median)':<26} MAE ${mean_absolute_error(y_test, country_pred):>9,.0f}\n")

# --- Cross-validation on the training set ----------------------------------
# Every fitted transformer inside the pipeline is refit per fold, so the
# imputer medians and encoder categories never see held-out rows.

for name, model in models.items():
    scores = -cross_val_score(
        model, X_train, y_train_log, cv=5,
        scoring='neg_mean_absolute_error'
    )
    print(f"{name:<26} CV log-MAE {scores.mean():.3f} +/- {scores.std():.3f}")

print()

# --- Final fit and holdout evaluation --------------------------------------

results = df.loc[X_test.index, ['Country', 'DevType', 'WorkExp']].copy()
results['actual'] = y_test

for name, model in models.items():
    model.fit(X_train, y_train_log)
    pred = np.exp(model.predict(X_test))
    results[f'pred_{name}'] = pred.round(0)
    print(f"{name:<26} test MAE ${mean_absolute_error(y_test, pred):>9,.0f}   "
          f"medAE ${np.median(np.abs(y_test - pred)):>9,.0f}   "
          f"R2 {r2_score(np.log(y_test), np.log(pred)):.3f}")

print(f"\nfeatures after encoding: "
      f"{len(models['gbm'].named_steps['prep'].get_feature_names_out())}")

# --- Inspection ------------------------------------------------------------

results['error'] = results.pred_gbm - results.actual
print("\nsample predictions")
print(results.head(10).to_string(index=False))

print("\nmean error by country (top 10 by count)")
by_country = results.groupby('Country')['error'].agg(['count', 'mean'])
print(by_country.nlargest(10, 'count').round(0).to_string())

results.to_csv('./data/predictions.csv', index=False)