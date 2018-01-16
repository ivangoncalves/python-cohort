# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset and use only columns 1 and 2. machine_id and dt.
df = pd.read_csv('device_by_date.csv',
                      usecols=[0,1],
                      infer_datetime_format = True)

# rename columns ...machineID and ...dt
df.columns = ['UserId', 'date']

# transform the date from text to date type. 
df['date'] = pd.to_datetime(df['date'])

# create a new column for Year-month only since we are agg by month 
df['OrderPeriod'] = df.date.apply(lambda x: x.strftime('%Y-%m'))

# Determine the user's cohort group (based on their first order)
df.set_index('UserId', inplace=True)
df['CohortGroup'] = df.groupby(level=0)['date'].min().apply(lambda x: x.strftime('%Y-%m'))
df.reset_index(inplace=True)

# . Rollup data by CohortGroup
grouped = df.groupby(['CohortGroup', 'OrderPeriod'])

# count unique Machine Ids. 
cohorts = grouped.agg({'UserId': pd.Series.nunique})
# change the name of the column.
cohorts.rename(columns={'UserId': 'TotalUsers'}, inplace = True)
#cohorts.head()

def cohort_period(df):
    df['CohortPeriod'] = np.arange(len(df)) + 1
    return df

cohorts = cohorts.groupby(level=0).apply(cohort_period)
#cohorts.head()

# reindex the DataFrame
cohorts.reset_index(inplace=True)
cohorts.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)

# create a Series holding the total size of each CohortGroup
cohort_group_size = cohorts['TotalUsers'].groupby(level=0).first()
cohort_group_size.head()

# shows with the months. 
cohorts['TotalUsers'].unstack(0).head()

# Montly cohort percentage
user_retention = cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)
user_retention.head(10)

# CHARTS 
sns.set(style='white')

plt.figure(figsize=(12, 8))
plt.title('Cohorts: User Retention')
sns_plot = sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%')

# Save cohort into an image 
image = sns_plot.get_figure()
image.savefig('cohort_image.png')
