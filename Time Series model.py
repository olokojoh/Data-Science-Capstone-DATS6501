import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import signal

#%%
#import data
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
plt.title('gold price per ounce in USD')
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
#it shows that gold price has high autocorrelation that we can apply
#time series model on it

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
stat =gold['price'].values
result = adfuller(stat)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#Now the dataset is stationary as p-value is about 0 which is lower than 0.05
#%%
#Make prediction on nonstationary dataset use exponential smoothing method
#split train,test
from sklearn.model_selection import train_test_split
Y=gold[['USD']]
y_train,y_test=train_test_split(Y,test_size=0.2,shuffle=False)
#%%
y_train.info()
#%%
y_train.head()


#%%
#use classical decomposition to decompose the data
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(gold['USD'], model='additive', freq=1)
result.plot()
plt.title('Addtive seasonal')
plt.show()

#%%
result1 = seasonal_decompose(gold['USD'], model='multiplicative',freq=1)
result1.plot()
plt.title('Multiplicative seasonal')
plt.show()

#%%
#Clearly, naive approach method and simple average method is not suitable in our case.
#Firstly, we try simple exponential smoothing method, pick the alpha value to be 0.5
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
fit2 = SimpleExpSmoothing(np.asarray(y_train['USD'])).fit(smoothing_level=0.5,optimized=False)
y_test['SES'] = fit2.forecast(len(y_test))
y_test.head()

#%%
ttt=np.array(y_test['USD'])
error_SES=ttt-y_test['SES'].values
acf=[]
for i in range(20):
    acf.append(get_auto_corr(error_SES,i))
L1=np.arange(0,20,1)
L2=-L1[::-1]
x = np.concatenate((L2[0:-1], L1))
acf_reverse = acf[::-1]
ACF = np.concatenate ((acf_reverse[0:-1], acf))
plt.stem(x,ACF, use_line_collection=True, markerfmt = 'o')
plt.xlabel('lags')
plt.ylabel('ACF value')
plt.title('ACF for 20 values SES model error')
plt.show()
#%%
acf.remove(acf[0])
acf1=np.array(acf)
Q_SES=len(error_SES)*np.sum(acf1**2)
var_SES=np.var(error_SES)
mse_SES=np.mean(error_SES**2)
mean_SES=np.mean(error_SES)
rmse_SES=(mse_SES)**0.5
print("The Q value of SES model is:",Q_SES)
print("The variance of SES model is:",var_SES)
print("The mse of SES model is:",mse_SES)
print("The mean of SES model error is:",mean_SES)
print("RMSE of SES model is:",rmse_SES)
#%%
#apply holt linear method to predict as from the decomposition graph, the gold price has increasing trend.
#I have tried other smoothing levels and slopes, and I found that smoothing-leel=0.3 and smoothing slope=0.1 is best
fit1 = Holt(np.asarray(y_train['USD'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_test['Holt_linear'] = fit1.forecast(len(y_test))
y_test.head()
#%%
ttt=np.array(y_test['USD'])
error_holt=ttt-y_test['Holt_linear'].values
acf=[]
for i in range(20):
    acf.append(get_auto_corr(error_holt,i))
L1=np.arange(0,20,1)
L2=-L1[::-1]
x = np.concatenate((L2[0:-1], L1))
acf_reverse = acf[::-1]
ACF = np.concatenate ((acf_reverse[0:-1], acf))
plt.stem(x,ACF, use_line_collection=True, markerfmt = 'o')
plt.xlabel('lags')
plt.ylabel('ACF value')
plt.title('ACF for 20 values Holt Linear model error')
plt.show()
#%%
acf.remove(acf[0])
acf1=np.array(acf)
Q_linear=len(error_holt)*np.sum(acf1**2)
var_linear=np.var(error_holt)
mse_linear=np.mean(error_holt**2)
mean_linear=np.mean(error_holt)
rmse_linear=(mse_linear)**0.5
print("The Q value of Holt Linear model is:",Q_linear)
print("The variance of Holt Linear model is:",var_linear)
print("The mse of Holt Linear is:",mse_linear)
print("The mean of Holt Linear error is:",mean_linear)
print("RMSE of Holt Linear error is:",rmse_linear)

#%%
#apply holt winter method to predict, as there are 12 years , so period chosen to be 12
#the whole trend of the data is increasing , so trend be add
#there is residual in the multiplicative decomposition and the seasonal variation seems be stationary
#so we choose the seasonl be additive as well
from statsmodels.tsa.api import ExponentialSmoothing
model =ExponentialSmoothing(np.asarray(y_train['USD']), seasonal_periods=12, trend='add', seasonal='add').fit(use_boxcox=True)
y_test['Holt_Winter'] = model.forecast(len(y_test))
y_test.head()

#%%
ttt=np.array(y_test['USD'])
error_winter=ttt-y_test['Holt_Winter'].values
acf=[]
for i in range(20):
    acf.append(get_auto_corr(error_winter,i))
L1=np.arange(0,20,1)
L2=-L1[::-1]
x = np.concatenate((L2[0:-1], L1))
acf_reverse = acf[::-1]
ACF = np.concatenate ((acf_reverse[0:-1], acf))
plt.stem(x,ACF, use_line_collection=True, markerfmt = 'o')
plt.xlabel('lags')
plt.ylabel('ACF value')
plt.title('ACF for 20 values Holt Winter model error')
plt.show()

#%%
acf.remove(acf[0])
acf1=np.array(acf)
Q_winter=len(error_winter)*np.sum(acf1**2)
var_winter=np.var(error_winter)
mse_winter=np.mean(error_winter**2)
mean_winter=np.mean(error_winter)
rmse_winter=(mse_winter)**0.5
print("The Q value of Holt Winter model is:",Q_winter)
print("The variance of Holt Winter model is:",var_winter)
print("The mse of Holt Winter model is:",mse_winter)
print("The mean of Holt Winter model error is:",mean_winter)
print("RMSE of Holt Winter error is:",rmse_winter)


#%%
plt.figure(figsize=(12,8))
plt.plot(y_train['USD'], label='Train',marker='o',markersize=4)
plt.plot(y_test['USD'], label='Test',marker='p',markersize=4)
plt.plot(y_test['Holt_linear'], label='Holt_linear',marker='+',markersize=4)
plt.plot(y_test['SES'],label='SES',marker='x',markersize=4)
plt.plot(y_test['Holt_Winter'],label='Holt_Winter',marker='2',markersize=4)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('USD price of gold')
plt.title('Plot of predicted value versus true values')
plt.show()

#%%
data={'method':['Holt Winter','Holt Linear','SES'],
      'MSE':[mse_winter,mse_linear,mse_SES],
      'mean':[mean_winter,mean_linear,mean_SES],
      'variance':[var_winter,var_linear,var_SES],
      'Q value':[Q_winter,Q_linear,Q_SES],
      'RMSE':[rmse_winter,rmse_linear,rmse_SES]}
table=pd.DataFrame(data)
table
