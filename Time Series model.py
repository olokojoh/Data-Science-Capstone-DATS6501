import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import signal
from Preprocessing import get_auto_corr

#%%
#import data
import random
gold = pd.read_csv('./data/gold.csv')

#%%
#plot the target value
plt.figure(figsize=(14,8))
plt.plot(gold['USD'])
plt.xlabel('date')
plt.ylabel('gold price')
plt.title('gold price($) per ounce per day')
plt.show()
#We can see from the plot that the dataset is not stationary, and the gold price has an increasing trend.

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
#detrended transformation
from stldecompose import decompose
decomp = decompose(gold['USD'], period=365)
decomp.plot()
plt.title('STL decomposition')
plt.show()
#%%
plt.figure(figsize=(14,8))
gold['diff']=gold['USD']-decomp.trend
plt.plot(gold['diff'])
plt.title('detrended series')
plt.xlabel('date')
plt.ylabel('gold price')
plt.show()
#%%
stat =gold['diff'].values
result = adfuller(stat)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#%%
#Make prediction on nonstationary dataset use exponential smoothing method
#split train,test
from sklearn.model_selection import train_test_split
Y=gold[['USD']]
y_train,y_test=train_test_split(Y,test_size=0.1,shuffle=False)
#%%
from statsmodels.tsa.api import Holt
#apply holt linear method to predict as from the decomposition graph, the gold price has increasing trend.
#I have tried other smoothing levels and slopes, and I found that smoothing-leel=0.3 and smoothing slope=0.1 is best
fit1 = Holt(np.asarray(y_train['USD'])).fit(smoothing_level = 0.1,smoothing_slope = 0.35)
y_test['Holt_linear'] = fit1.forecast(len(y_test))
y_test.head()
#%%
plt.plot(y_test['USD'],label="test")
plt.plot(y_test['Holt_linear'],label="Holt linear")
plt.legend(loc="best")
plt.title('Holt linear prediction')
plt.xlabel('date')
plt.ylabel('gold price')
plt.show()
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
plt.title('ACF with 20 lags of Holt Linear model error')
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
#Winter
from statsmodels.tsa.api import ExponentialSmoothing
model =ExponentialSmoothing(np.asarray(y_train['USD']), seasonal_periods=12, trend='add', seasonal='add').fit(use_boxcox=True)
y_test['Holt_Winter'] = model.forecast(len(y_test))
y_test.head()
#%%
plt.plot(y_test['USD'],label="test")
plt.plot(y_test['Holt_Winter'],label="Holt Winter")
plt.legend(loc="best")
plt.title('Holt Winter prediction')
plt.xlabel('date')
plt.ylabel('gold price')
plt.show()


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
#SES
from statsmodels.tsa.api import SimpleExpSmoothing
Y=gold[['diff']]
yy_train,yy_test=train_test_split(Y,test_size=0.1,shuffle=False)
fit3 = SimpleExpSmoothing(np.asarray(yy_train['diff'])).fit(smoothing_level=0.5,optimized=False)
yy_test['SES'] = fit3.forecast(len(yy_test))
yy_test.head()
#%%
plt.plot(yy_test['diff'],label="test")
plt.plot(yy_test['SES'],label="SES")
plt.legend(loc="best")
plt.xlabel('date')
plt.ylabel('gold price')
plt.title('SES prediction')
plt.show()
#%%
ttt=np.array(yy_test['diff'])
error_SES=ttt-yy_test['SES'].values
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
plt.title('ACF with 20 lags of SES model error')
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
#ARMA
#remind of ADF test
stat =gold['diff'].values
result = adfuller(stat)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#%%
#build GPAC table to determine the AR and MA order
y=np.array(yy_train['diff'])
acf=[]
for i in range(100):
    acf.append(get_auto_corr(y,i+1))
ry=[np.var(y)]
for i in range(99):
    ry.append(acf[i+1]*np.var(y))
#%%
phi=[]
phi_1=[]
i=0
gpac = np.zeros(shape=(8, 7))
for j in range(0,8):
    for k in range(2,9):
        bottom = np.zeros(shape=(k, k))
        top = np.zeros(shape=(k, k))
        for m in range(k):
            for n in range(k):
                bottom[m][n]=ry[abs(j+m - n)]
            top[m][-1]=ry[abs(j+m+1)]
        i=i+1
        top[:,:k-1] = bottom[:,:k-1]
        phi.append(round((np.linalg.det(top) / np.linalg.det(bottom)),2))
    phi_1.append(round(ry[j + 1] / ry[j],2))
gpac=np.array(phi).reshape(8,7)
Phi1=pd.DataFrame(phi_1)
Gpac=pd.DataFrame(gpac)
GPAC = pd.concat([Phi1,Gpac], axis=1)
GPAC.columns=['1','2','3','4','5','6','7','8']
print(GPAC)

#%%
#use heatmap to see the table clearly
import seaborn as sns
sns.heatmap(GPAC, center=0, annot=True)
plt.title("Generalized partial autocorrelation function ")
plt.xlabel("na")
plt.ylabel("nb")
plt.show()

#%%
#na=2, nb=1
model1=sm.tsa.ARMA(y,(2,1)).fit(trend='nc',disp=0)
print(model1.summary())
#%%
print("The confidence interval of ARMA(2,1) model is:",model1.conf_int(alpha=0.05, cols=None))
print("The covariance matrix of ARMA(2,1) model is:",model1.cov_params())

#%%
print(len(yy_test))
#%%
result = model1.predict(start=1,end=1334)
true=np.array(yy_test['diff'])
error_21=true-result
yy_test['ARMA21']=result
yy_test.tail()
#%%
plt.plot(yy_test['diff'],label="test")
plt.plot(yy_test['ARMA21'],label="ARMA21")
plt.legend(loc="best")
plt.title('ARMA(2,1) prediction')
plt.xlabel('date')
plt.ylabel('trend difference')
plt.show()

#%%
acf=[]
for i in range(20):
    acf.append(get_auto_corr(error_21,i))
L1=np.arange(0,20,1)
L2=-L1[::-1]
x = np.concatenate((L2[0:-1], L1))
acf_reverse = acf[::-1]
ACF = np.concatenate ((acf_reverse[0:-1], acf))
plt.stem(x,ACF, use_line_collection=True, markerfmt = 'o')
plt.xlabel('lags')
plt.ylabel('ACF value')
plt.title('ACF with 20 lags of ARMA(2,1) model error')
plt.show()

#%%
acf.remove(acf[0])
acf1=np.array(acf)
Q21=len(error_21)*np.sum(acf1**2)
var_21=np.var(error_21)
mse_21=np.mean(error_21**2)
mean_21=np.mean(error_21)
rmse_21=(mse_21)**0.5
print("The Q value of ARMA(2,1) is:",Q21)
print("The variance of ARMA(2,1) is:",var_21)
print("The mse of ARMA(2,1) is:",mse_21)
print("The mean of ARMA(2,1) error is:",mean_21)
print("RMSE of ARMA(2,1) error is:",rmse_21)

#%%
gold['trend']=decomp.trend
tr=gold['trend']
x_train,x_test=train_test_split(tr,test_size=0.1,shuffle=False)
y_test['ARMA21']=x_test+yy_test['ARMA21']
y_test['SES']=x_test+yy_test['SES']
y_test.head()
#%%
#take a look
plt.figure(figsize=(12,8))
plt.plot(y_test['ARMA21'], label='ARMA21',marker='o',markersize=2)
plt.plot(y_test['USD'], label='True',marker='p',markersize=2)
plt.plot(y_test['Holt_linear'], label='Holt_linear',marker='+',markersize=2)
plt.plot(y_test['SES'],label='SES',marker='x',markersize=2)
plt.plot(y_test['Holt_Winter'],label='Holt_Winter',marker='2',markersize=2)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('USD price of gold')
plt.title('Plot of predicted value versus true values')
plt.show()

#%%
data={'method':['Holt Winter','Holt Linear','SES','ARMA(2,1)'],
      'MSE':[mse_winter,mse_linear,mse_SES,mse_21],
      'mean':[mean_winter,mean_linear,mean_SES,mean_21],
      'variance':[var_winter,var_linear,var_SES,var_21],
      'Q value':[Q_winter,Q_linear,Q_SES,Q21],
      'RMSE':[rmse_winter,rmse_linear,rmse_SES,rmse_21]}
table=pd.DataFrame(data)
table
