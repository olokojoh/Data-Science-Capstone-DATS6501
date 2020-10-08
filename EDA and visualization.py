import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import signal

#%%
#import data
import random
random.seed(42)
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)
df=pd.read_csv('LBMA-GOLD.csv')
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

#%%
gold.head()

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

#%%
#plot the target value
plt.plot(gold['USD'])
plt.xlabel('date')
plt.ylabel('gold price')
plt.title('gold price($) per ounce per day')
plt.show()
#We can see from the plot that the dataset is not stationary, and the gold price has an increasing trend.
#%%
#plot the autocorrelation of the gold price
dep=np.array(gold['USD'])
acf=[]
for i in range(20):
    acf.append(get_auto_corr(dep,i))
L1=np.arange(0,20,1)
L2=-L1[::-1]
x = np.concatenate((L2[0:-1], L1))
acf_reverse = acf[::-1]
ACF = np.concatenate ((acf_reverse[0:-1], acf))
plt.stem(x,ACF, use_line_collection=True, markerfmt = 'o')
plt.xlabel('lags')
plt.ylabel('ACF value')
plt.title('ACF for USD price')
plt.show()
#Almost perfect positive correlation

#%%
#from the gold price plot, we can see that the mean and variance of the
#gold price is not stationary. To make time series model. firstly, we need
#to make the gold price stationary. We can apply ADF test to see if the
#the dataset if stationary.
from statsmodels.tsa.stattools import adfuller
stat =gold['USD'].values
result = adfuller(stat)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#the p-value is 0.995 which is higher than 0.05

#%%
#try first difference transformation method
#y(i)=y(t)-y(t-1)
gold['price']=(gold['USD']-gold['USD'].shift(1)).dropna()
gold=gold.drop(gold.index[0])
gold.head(5)
#%%
gold.info()
#%%
stat =gold['price'].values
result = adfuller(stat)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#Now the dataset is stationary as p-value is about 0 which is lower than 0.05

#%%
plt.plot(gold['price'])
plt.xlabel('date')
plt.ylabel('first difference')
plt.title('first difference per day')
plt.show()
