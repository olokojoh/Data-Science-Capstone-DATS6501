import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import signal
from Preprocessing import get_auto_corr

#%%
# import data
gold = pd.read_csv('./data/gold.csv')

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
