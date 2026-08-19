# Data Cleanup Tasks
By Aarish Kodnaney
- Remove all rows where "annual_salary_usd" is 0 or NaN
- Remove all rows where "Employment" is equal to any of the following: ["Retired", "Not employed", "Student", "I prefer not to say"]
- Remove all rows where countr is equal to any of the following: ['United States of America', 'Australia', 'Sweden', 'Denmark', 'Norway', 'Switzerland', 'Germany', 'United Kingdom of Great Britain and Northern Ireland'] **where** `1000 <= annual_salary_usd <= 15,000` **and** `Employment != Independent contractor, freelancer, or self-employed` - It is plausible that a freelancer could be making this salary, but if an employer person is making that much, they are either an intern or this is a protest answer. 
- An imputer will be used for NaN values in `WorkExp` - that imputer will be the median
- With .4% being NaN in `YearsCode` we will also use a median imputer there
- We will drop situations where `YearsCode` has more coding years than the age bracket allows
- I considered dropping situations where `WorkExp > YearsCode` but it could be plausible that somebody made a mid career switch and has been coding for less time than they've worked in total
- We will drop all rows with an annual salary below $1000 - this doesn't make sense no matter the location. It's likely someone accidentally input their hourly or weekly pay, but we cannot read their mind to be sure, so accuracy will probably be higher without this data
- Drop where `DevType == Student`

## Quantifying non-numerical values
- Age is currently represented by ranges. To quantify it I'm going to use the mean age of the range
- Since we only have `Employment` values of `["Employed", "Independent contractor, freelancer, or self-employed"]` we will represent them with 1 = independent contractor and 0 = employed
- Utilize a OneHotEncoder on DevType - there are a lot of values
- For OrgSize, we will first replace each text label with the approximate middle of the range it describes and then log transform that value 
- IDK maps to None which becomes NaN which will be handled by SimpleImputer(strategy='median)
- Utilize binary encoding for `ICorPM`
- Utilize OneHotEncoder for `RemoteWork` with every categoty being included along with na being it's on category (na is 11% of total responses for this column)
- Utilize OneHotEncoder for `Industry` with `fill_value='Missing'` for the 2.88% of na values
- To quantify country, I would like to use OneHotEncoder, but there are 130 unique countries listed in the dataset. Instead, I'm going to map each country to a world bank tier and then use OneHotEncoder
- I'm dropping Currency. There's 97 currencies people are being paid in and it is near collinear with country. The only nuance is a developer in poland or russia being paid in USD is likely working remotely for a foreign company, meaning that they are being paid significantly more than a standard polish developer. A valid feature I would implement would be passing a 1 if `currency != expected_local_currency` but that would require actual mapping of country to currency. I'll add this at the end if I have extra time
- For now, both `LanguagesHaveWorkedWith` and `DatabaseHaveWorkedWith` will be quantified by the count of semicolons + 1. If the value is null, then it will be quantified by 0

EdLevel level will be quantified in the following manner:

| Education Level | Value |
|---|---|
| Primary/elementary school | 0 |
| Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.) | 1 |
| Some college/university study without earning a degree | 2 |
| Associate degree (A.A., A.S., etc.) | 3 |
| Bachelor's degree (B.A., B.S., B.Eng., etc.) | 4 |
| Master's degree (M.A., M.S., MBA, M.Eng., etc.) | 5 |
| Professional degree (JD, MD, Ph.D, Ed.D, etc.) | 5 |
| Other (please specify): | None |