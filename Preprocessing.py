import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import signal

#%%
#import data
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)
df=pd.read_csv('gold.csv')
df.head(5)

#%%
##EDA and preprocessing

#%%
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

