import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import signal
import os.path
from os import path

#%%
#import data
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)
df=pd.read_csv('./data/LBMA-GOLD.csv')
df.head(5)
df.shape
#%%
##EDA and preprocessing

#check missing value
def nan_checker(df):
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'proportion', 'dtype'])
    df_nan = df_nan.sort_values(by='proportion', ascending=False)
    return df_nan
df_nan = nan_checker(df)
df_nan.reset_index(drop=True)

#%%
#sort in date in ascending order
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.sort_values(by='Date', inplace=True, ascending=True)
df.head(5)

#%%
#change the index to be the date
df.index = df['Date']
df.drop('Date',axis = 1, inplace = True)
df.head(5)
#%%
#pick usd(am) as target value and create a new dataframe
df.columns.values[0] = 'USD'
df.head(5)

#%%
gold = df.iloc[: , [0]].copy()
gold.head(5)
#%%
#see how many missing values
gold.isnull().sum()
#1 missing value
#%%
gold.info()
#%%
gold.dropna(inplace=True)
gold.info()
#%%
df_nan = nan_checker(gold)
df_nan.reset_index(drop=True)

# %%
gold.info()
#%%
def get_auto_corr(timeSeries,k):
    l = len(timeSeries)
    timeSeries1 = timeSeries[0:l-k]
    timeSeries2 = timeSeries[k:]
    timeSeries_mean = np.mean(timeSeries)
    timeSeries_var = np.array([i**2 for i in timeSeries-timeSeries_mean]).sum()
    auto_corr = 0
    for i in range(l-k):
        temp = (timeSeries1[i]-timeSeries_mean)*(timeSeries2[i]-timeSeries_mean)/timeSeries_var
        auto_corr = auto_corr + temp
    return auto_corr

# %%
# save file to csv
goldf_path = './data/gold.csv'
if not os.path.isfile(goldf_path):
    gold.to_csv(goldf_path)
    print(goldf_path + ' is created!')
# %%
