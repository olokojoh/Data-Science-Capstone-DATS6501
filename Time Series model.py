#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import signal
from Preprocessing import get_auto_corr

#%%
#import data

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
plt.plot(y_test['USD'],label="test")
plt.plot(y_test['SES'],label="SES")
plt.legend(loc="best")
plt.xlabel('date')
plt.ylabel('gold price')
plt.title('SES prediction')
plt.show()
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
#apply holt linear method to predict as from the decomposition graph, the gold price has increasing trend.
#I have tried other smoothing levels and slopes, and I found that smoothing-leel=0.3 and smoothing slope=0.1 is best
fit1 = Holt(np.asarray(y_train['USD'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
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
#apply holt winter method to predict, choose period=365
#the whole trend of the data is increasing , so trend be add
#there is residual in the multiplicative decomposition and the seasonal variation does not change a lot in proportion
#so we choose the seasonl be additive as well
from statsmodels.tsa.api import ExponentialSmoothing
model =ExponentialSmoothing(np.asarray(y_train['USD']), seasonal_periods=365, trend='add', seasonal='add').fit(use_boxcox=True)
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
plt.figure(figsize=(12,8))
plt.plot(y_train['USD'], label='Train',marker='o',markersize=2)
plt.plot(y_test['USD'], label='Test',marker='p',markersize=2)
plt.plot(y_test['Holt_linear'], label='Holt_linear',marker='+',markersize=2)
plt.plot(y_test['SES'],label='SES',marker='x',markersize=2)
plt.plot(y_test['Holt_Winter'],label='Holt_Winter',marker='2',markersize=2)
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


#%%
#ARMA
#%%
gold.head()


#%%
plt.plot(gold['price'])
plt.xlabel('date')
plt.ylabel('first difference')
plt.title('first difference data plot')
plt.show()

#%%
#ACF plot of the first difference data
dep=np.array(gold['price'])
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
plt.title('ACF for first difference')
plt.show()

#%%
#remind of ADF test
stat =gold['price'].values
result = adfuller(stat)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#stationary data

#%%
#split train, test
Y=gold[['price']]
yy_train,yy_test=train_test_split(Y,test_size=0.2,shuffle=False)

#%%
yy_test.head()
#%%
#build GPAC table to determine the AR and MA order
y=np.array(yy_train['price'])
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
#pass zero/pole cancellation
np.roots([1,-0.6,0])
#%%
np.roots([1,0.6,0])

#%%
print("The confidence interval of ARMA(2,1) model is:",model1.conf_int(alpha=0.05, cols=None))
print("The covariance matrix of ARMA(2,1) model is:",model1.cov_params())

#%%
print(len(yy_test))
#%%
result = model1.predict(start=0,end=2666)
true=np.array(yy_test['price'])
error_21=true-result
yy_test['ARMA21']=result
yy_test.head()
#%%
plt.plot(yy_test['price'],label="test")
plt.plot(yy_test['ARMA21'],label="ARMA21")
plt.legend(loc="best")
plt.title('ARMA(2,1) prediction')
plt.xlabel('date')
plt.ylabel('first difference')
plt.show()

#%%
acf=[]
for i in range(100):
    acf.append(get_auto_corr(error_21,i))
L1=np.arange(0,100,1)
L2=-L1[::-1]
x = np.concatenate((L2[0:-1], L1))
acf_reverse = acf[::-1]
ACF = np.concatenate ((acf_reverse[0:-1], acf))
plt.stem(x,ACF, use_line_collection=True, markerfmt = 'o')
plt.xlabel('lags')
plt.ylabel('ACF value')
plt.title('ACF with 100 lags of ARMA(2,1) model error')
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
#Whether pass the white test
from scipy.stats import chi2
DOF=100-3
alfa=0.01
chi_critical=chi2.ppf(1-alfa,DOF)
if Q21<chi_critical:
    print("the residual is white")
else:
    print("Not white")

#%%
#na=2,nb=2
model2=sm.tsa.ARMA(y,(2,2)).fit(trend='nc',disp=0)
print(model2.summary())
#%%
#pass zero/pole cancellation
np.roots([1,-0.7,-0.6])
#fail to pass

#%%
#try na=3,nb=1
model3=sm.tsa.ARMA(y,(3,1)).fit(trend='nc',disp=0)
print(model3.summary())
#%%
#pass zero/pole cancellation
np.roots([1,-0.15,0,0])
#%%
np.roots([1,0.1,0,0])
#pass

#%%
print("The confidence interval of ARMA(3,1) model is:",model3.conf_int(alpha=0.05, cols=None))
print("The covariance matrix of ARMA(3,1) model is:",model3.cov_params())

#%%
#make prediction
result2 = model3.predict(start=0,end=2666)
error_31=true-result2
yy_test['ARMA31']=result2
yy_test.head()

#%%
plt.plot(yy_test['price'],label="test")
plt.plot(yy_test['ARMA31'],label="ARMA31")
plt.legend(loc="best")
plt.title('ARMA(3,1) prediction')
plt.xlabel('date')
plt.ylabel('first difference')
plt.show()

#%%
acf=[]
for i in range(100):
    acf.append(get_auto_corr(error_31,i))
L1=np.arange(0,100,1)
L2=-L1[::-1]
x = np.concatenate((L2[0:-1], L1))
acf_reverse = acf[::-1]
ACF = np.concatenate ((acf_reverse[0:-1], acf))
plt.stem(x,ACF, use_line_collection=True, markerfmt = 'o')
plt.xlabel('lags')
plt.ylabel('ACF value')
plt.title('ACF with 100 lags of ARMA(3,1) model error')
plt.show()

#%%
acf.remove(acf[0])
acf1=np.array(acf)
Q31=len(error_31)*np.sum(acf1**2)
var_31=np.var(error_31)
mse_31=np.mean(error_31**2)
mean_31=np.mean(error_31)
rmse_31=(mse_31)**0.5
print("The Q value of ARMA(3,1) is:",Q31)
print("The variance of ARMA(3,1) is:",var_31)
print("The mse of ARMA(3,1) is:",mse_31)
print("The mean of ARMA(3,1) error is:",mean_31)
print("RMSE of ARMA(3,1) error is:",rmse_31)

#%%
from scipy.stats import chi2
DOF=100-4
alfa=0.01
chi_critical=chi2.ppf(1-alfa,DOF)
if Q31<chi_critical:
    print("the residual is white")
else:
    print("Not white")
#pass white noise test

#%%
#make compasrison
data={'method':['ARMA(2,1)','ARMA(3,1)'],
      'MSE':[mse_21,mse_31],
      'mean':[mean_21,mean_31],
      'variance':[var_21,var_31],
      'Q value':[Q21,Q31],
      'RMSE':[rmse_21,rmse_31]}
table=pd.DataFrame(data)
table
#%%
yy_test.head()
#%%
plt.figure(figsize=(10,8))
plt.plot(yy_test['price'], label='test',marker='o',markersize=2)
plt.plot(yy_test['ARMA21'], label='ARMA21',marker='p',markersize=2)
plt.plot(yy_test['ARMA31'],label='ARMA31',marker='+',markersize=2)
plt.plot(yy_train['price'],label='train',marker='x',markersize=2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Plot of predicted value versus true values')
plt.legend(loc='best')
plt.show()

#%%
#The two ARMA models performance are almost same, ARMA(3,1) perform little better
#with less mean error and less MSE and RMSE
#So I finally pick ARMA(3,1) model
#Now add the values back to the raw dataset
diff=np.array(yy_test['ARMA31'])
raw=np.array(y_test['USD'])
list=[raw[0]]
for i in range(1,2667):
    c=raw[i-1]+diff[i]
    list.append(c)
y_test['ARMA31']=list
y_test.tail()
#%%
#take a look
plt.figure(figsize=(12,8))
plt.plot(y_test['ARMA31'], label='ARMA31',marker='o',markersize=6)
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
data={'method':['Holt Winter','Holt Linear','SES','ARMA(3,1)'],
      'MSE':[mse_winter,mse_linear,mse_SES,mse_31],
      'mean':[mean_winter,mean_linear,mean_SES,mean_31],
      'variance':[var_winter,var_linear,var_SES,var_31],
      'Q value':[Q_winter,Q_linear,Q_SES,Q31],
      'RMSE':[rmse_winter,rmse_linear,rmse_SES,rmse_31]}
table=pd.DataFrame(data)
table
