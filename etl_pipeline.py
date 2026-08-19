"""
By: Aarish Kodnaney
For: Generate F26 Data Engineer Application
etl_pipeline.py
"""

from typing import Optional
import pandas as pd
from pandas import DataFrame, Series
import wbgapi as wb
import numpy as np

df: DataFrame = pd.read_csv('./data/survey.csv')
print("sucessfully loaded data from file: survey.csv")

# remove all rows where annual_salary_usd is na
before: int = len(df)
df = df[df.annual_salary_usd.notna()]
assert df['annual_salary_usd'].isna().sum() == 0
print(f"sucessfully removed na values from dataframe | removed {before - len(df)} rows")

# conversion of countries to World Bank ISO3 codes
codes = {c: wb.economy.coder(c) for c in df["Country"].dropna().unique()}
df["Country"] = df["Country"].map(codes)
tier_lookup: dict[str, str] = wb.economy.DataFrame()["incomeLevel"].to_dict()
df["income_tier"] = df["Country"].map(tier_lookup.get)
print("succesfully converted countries to WBISO3 codes")

# drop rows where annual_salary_usd <= 1000 AND income_tier is not 'LIC'
before = len(df)
df = df[(df.annual_salary_usd > 1000) | (df.income_tier == "LIC")] # including places where income_tier == LIC kept 3 rows where this salary is feasible
assert not ((df.annual_salary_usd <= 1000) & (df.income_tier != "LIC")).any()
assert len(df) > 0, "filter removed every row"
removed = before - len(df)
print(f"removed {removed} rows (low salary, non-LIC) | {len(df)} remain")

# remove all rows where Employment type is not in allowed - I'm not tracking students or unemployed devs
before = len(df)
allowed = ["Employed", "Independent contractor, freelancer, or self-employed"]
df = df[df.Employment.isin(allowed)]
assert df["Employment"].isin(allowed).all()
print(f"eliminated all responses with employment type that is not ['Employed', 'Independent contractor, freelancer, or self-employed'] | removed {before - len(df)} rows")

# remove all rows where income tier is HIC (high income), 1000 <= annual_salary_usd <= 15000, and employment type is not 'Independent contractor, freelancer, or self-employed'
before = len(df)
rows_to_remove: Series = ((df.annual_salary_usd.between(1000, 15000)) & (df.income_tier == "HIC") & (df.Employment != 'Independent contractor, freelancer, or self-employed'))
df = df[~rows_to_remove]
assert not ((df.annual_salary_usd.between(1000, 15000)) & (df.income_tier == "HIC") & (df.Employment != 'Independent contractor, freelancer, or self-employed')).any() # we repeat this because variable specs is based on the old df
print(f"removed rows where income tier == 'HIC' and 1000 <= df.annual_salary_usd <= 15000, and df.Employment != 'Independent contractor, freelancer, or self-employed' | removed {before - len(df)} rows")

# drop situations where YearsCode > age range Age. Ignore NaN for now
age_max = {
    '18-24 years old': 24,
    '25-34 years old': 34,
    '35-44 years old': 44,
    '45-54 years old': 54,
    '55-64 years old': 64,
    '65 years or older': 90,
    'Prefer not to say': None,
}
before = len(df)
df['age_max'] = df.Age.map(age_max)
rows_to_remove = df.YearsCode > df.age_max
df = df[~rows_to_remove]
print(f"removed rows where YearsCode > top of Age range | removed {before - len(df)} rows")
assert not (df.age_max < df.YearsCode).any()

# drop situations where Work Exp > age range. Ignore NaN for now
before = len(df)
rows_to_remove = df.WorkExp > df.age_max
df = df[~rows_to_remove]
assert not (df.age_max < df.WorkExp).any()
print(f"removed rows where WorkExp > age_max | removed {before - len(df)} rows")

# drop where DevType is equal to 'Student'
before = len(df)
df = df[df["DevType"] != 'Student']
assert not (df.DevType == 'Student').any()
print("removed rows where DevType == 'Student' | removed {before - len(df)} rows")

# drop columns that are unecessary
df = df.drop(columns=['ResponseId', 'Currency'])

# ---------------------------------------------------------------------
# Quantifying non-numerical values
# ---------------------------------------------------------------------

# quantifying age ranges by using the mean value of the range
age_mean: dict[str, Optional[float]] = {
    '18-24 years old': 21,
    '25-34 years old': 29.5,
    '35-44 years old': 39.5,
    '45-54 years old': 49.5,
    '55-64 years old': 59.5,
    '65 years or older': 77.5,
    'Prefer not to say': None,
}
df['Age'] = df['Age'].map(age_mean)
print("quantified age column mean value of range")

# quantifying Employment value as 1 = independent contractor, 0 = employed, as these are the only two values

employment_quant: dict[str, int] = {
    'Employed': 0,
    'Independent contractor, freelancer, or self-employed': 1
}
df['Employment'] = df['Employment'].map(employment_quant)
print('quantified employment')


# quantifying ICorPM for 1 = PM, 0 = IC, NaN handled downstream
icorpm_quant: dict[str, int] = {
    'Individual contributor': 0,
    'People manager': 1
}
df['ICorPM'] = df['ICorPM'].map(icorpm_quant)
print("quantified ICorPM")

# quantifying OrgSize
orgsize_mid = {
    'Just me - I am a freelancer, sole proprietor, etc.': 1,
    'Less than 20 employees': 10,
    '20 to 99 employees': 59.5,
    '100 to 499 employees': 299.5,
    '500 to 999 employees': 749.5,
    '1,000 to 4,999 employees': 2999.5,
    '5,000 to 9,999 employees': 7499.5,
    '10,000 or more employees': 20000,
    'I don’t know': None,
}
unmapped = set(df.OrgSize.dropna().unique()) - set(orgsize_mid)
assert not unmapped, f"unmapped OrgSize labels: {unmapped}"
df['orgsize_log'] = np.log(df.OrgSize.map(orgsize_mid))
print("quantified orgSize")

# quantifying both of the haveWorkedWith columns
df['LanguageHaveWorkedWith'] = df.LanguageHaveWorkedWith.str.count(';').add(1)
df['DatabaseHaveWorkedWith'] = df.DatabaseHaveWorkedWith.str.count(';').add(1)
print('quantified both LanguageHaveWorkedWith and DatabaseHaveWorkedWith')

# quantifying edlevel
ed_map = {
    'Primary/elementary school': 0,
    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 1,
    'Some college/university study without earning a degree': 2,
    'Associate degree (A.A., A.S., etc.)': 3,
    'Bachelor\u2019s degree (B.A., B.S., B.Eng., etc.)': 4,
    'Master\u2019s degree (M.A., M.S., M.Eng., MBA, etc.)': 5,
    'Professional degree (JD, MD, Ph.D, Ed.D, etc.)': 6,
    'Other (please specify):': None,
}
unmapped = set(df.EdLevel.dropna().unique()) - set(ed_map)
assert not unmapped, f"unmapped EdLevel labels: {unmapped}"
df['EdLevel'] = df.EdLevel.map(ed_map)
print("quantified EdLevel")

# last min decision to quantify income_tier
tier_map = {'LIC': 0, 'LMC': 1, 'UMC': 2, 'HIC': 3}
df['income_tier'] = df.income_tier.map(tier_map)
print('quantified income_tier')

df = df.drop(columns='age_max') # this was just used to calculate another age
df = df.drop(columns='OrgSize') # superceded by orgsize_log
# df = df.drop(columns='Country') # data from income_tier will be used instead

print(f"pipeline produced dataframe with {len(df)} rows")

# save to parquet filetype for loading in model.py
df.to_csv('output.csv', index=False)
df.to_parquet('./data/clean.parquet', index=False)