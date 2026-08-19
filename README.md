# Aarish Kodnaney - Generate Data Technical Challenge

Findings can be viewed [here](https://northeastern-my.sharepoint.com/:p:/g/personal/kodnaney_a_northeastern_edu/IQBA4rRbP1QyQ5P5pzOk-jOXATWxk7BUaE5iLu6gLdC9HmI?e=cBLj6B). Must be logged into Northeastern account

## Steps I took
1. Familiarized myself with the dataset by just running code to get a feel for the data - [data_investigation.ipynb](./data_investigation.ipynb)
2. Wrote [data_cleanup_tasks.md](./data_cleanup_tasks.md) outlining what I wanted to change about the dataset. Refer to [this section](#Changes-I-made-to-the-data) to learn more
3. Wrote [etl_pipeline.py](./etl_pipeline.py) implementing the steps outlined in [data_cleanup_tasks.md](./data_cleanup_tasks.md). This didn't implement all steps, only the removal of bad or absurd situations/data in the dataset. It also quantified all non-numerical values that weren't categorical and dropped unecessary rows
4. Started [model.py](./model.py) by loading data created by [etl_pipeline.py](./etl_pipeline.py), splitting the data into 80% train and 20% test, imputing NaN values, and one-hot encoding categorical variables
5. Established my baseline being the median of the annual salaries. For any model to be useful it must beat this. 
6. Built two models, those being multiple regression and a histogram based gradient boosting regressor

## My understanding of models I used

### Multiple Regression
Basically just finding the equation resulting in a line that minimizes total squared error accross all 3800 trainiing rows

I however, didn't use standard Multiple Regression. I used Ridge, which basically means the model accepts slightly worse training fit in exchange for slightly smaller coefficients. This is important when dealing with salaries, where there can be shockingly high salaries that could screw up predictions for everything else. The other thing I did to account for this was taking the log of the training data to quantify error in percentage instead of dollar value. Then, I took the inverse log on my final predictions to convert back to real salary

### Histogram Based Gradient Boosting Regressor

Standard gradient boosting is a long line of guessuers, each one fixing the leftover mistakes of the one before.
When you make it Histogram based, you split all numerical values into 255 buckets and do some wizardy to make it happen faster


## Changes I made to the data

#### Based on [data_cleanup_tasks.md](./data_cleanup_tasks.md)

### Data Cleanup Tasks
- Remove all rows where "annual_salary_usd" is 0 or NaN
- Remove all rows where "Employment" is equal to any of the following: ["Retired", "Not employed", "Student", "I prefer not to say"]
- Remove all rows where countr is equal to any of the following: ['United States of America', 'Australia', 'Sweden', 'Denmark', 'Norway', 'Switzerland', 'Germany', 'United Kingdom of Great Britain and Northern Ireland'] **where** `1000 <= annual_salary_usd <= 15,000` **and** `Employment != Independent contractor, freelancer, or self-employed` - It is plausible that a freelancer could be making this salary, but if an employer person is making that much, they are either an intern or this is a protest answer. 
- An imputer will be used for NaN values in `WorkExp` - that imputer will be the median
- With .4% being NaN in `YearsCode` we will also use a median imputer there
- We will drop situations where `YearsCode` has more coding years than the age bracket allows
- I considered dropping situations where `WorkExp > YearsCode` but it could be plausible that somebody made a mid career switch and has been coding for less time than they've worked in total
- We will drop all rows with an annual salary below $1000 - this doesn't make sense no matter the location. It's likely someone accidentally input their hourly or weekly pay, but we cannot read their mind to be sure, so accuracy will probably be higher without this data
- Drop where `DevType == Student`


### Quantifying non-numerical values
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
