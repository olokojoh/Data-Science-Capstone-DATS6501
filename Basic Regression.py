# %%
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
gold = pd.read_csv('./data/gold.csv')
gold = gold.set_index(gold.Date)
gold.head()

# %%
# plot
plt.plot(gold['USD'])
plt.xlabel('date')
plt.ylabel('gold price')
plt.title('gold price per ounce in USD')
plt.show()
# %%
# Use moving average for past 3 days and 9 days as explanatory variable
gold['S_3'] = gold['USD'].shift(1).rolling(window=3).mean() 
gold['S_9']= gold['USD'].shift(1).rolling(window=9).mean() 
gold= gold.dropna() 
X = gold[['S_3','S_9']] 
X.head()
y = gold['USD']
y.head()

# %%
t=.8 
t = int(t*len(gold)) 
# Train dataset 
X_train = X[:t] 
y_train = y[:t]  
# Test dataset 
X_test = X[t:] 
y_test = y[t:]

# %%
linear = LinearRegression().fit(X_train,y_train) 
print ("Gold ETF Price =", round(linear.coef_[0],2),\
    "* 3 Days Moving Average", round(linear.coef_[1],2),\
    "* 9 Days Moving Average +", round(linear.intercept_,2))

predicted_price = linear.predict(X_test)  
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])  
predicted_price.plot(figsize=(10,5))  
y_test.plot()  
plt.legend(['predicted_price','actual_price'])  
plt.ylabel("Gold ETF Price")  
plt.show()

# %%
r2_score = linear.score(X[t:],y[t:])*100  
float("{0:.2f}".format(r2_score))