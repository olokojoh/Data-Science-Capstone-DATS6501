# %%
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential  #Sequential to initinal neural network
from keras.layers import Dense       #Dense to add full connect neural network
from keras.layers import LSTM        #LSTM to add long short term layer
from keras.layers import Dropout     #Dropout

# %%
import pandas as pd
import numpy as np

# Read csv file
df = pd.read_csv('./data/LBMA-GOLD.csv', header=0)
df.head()
# %%
from Preprocessing import nan_checker
# Select columns of USD
# Here I choose price of AM and PM to train the LSTM
df = df[['Date','USD (AM)','USD (PM)']]
# Check dataset
df.shape
# Here we have 13335 rows with Na values
# %%
# Drop Na value
df = df.dropna()
df.shape

# Now the data have 13193 rows without Na values
# %%

df.shape
df.reset_index(drop=True, inplace=True)
# %%
df.shape
df.reset_index(drop=True, inplace=True)
# %%
t = 0.7
train = df.iloc[:int(t * len(df)),:]
test = df.iloc[int(t * len(df)):,:]

#%%
train

train.iloc[:, 1:3].values

#%%
# Get training data
def train_data(PastnumDays, dataset_train):
    training_set = dataset_train.iloc[:, 1:3].values
    dataset_train.head()
        
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    
    X_train = []
    y_train = []
    for i in range(PastnumDays, len(dataset_train)):
        X_train.append(training_set_scaled[i-PastnumDays:i, :])
        y_train.append(training_set_scaled[i, :])

    X_train, y_train = np.array(X_train), np.array(y_train)      
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))
    
    return X_train,y_train,dataset_train,sc

#%%
# Get test data
def test_data(PastnumDays, dataset_train, dataset_test, sc):
    real_gold_price = dataset_test.iloc[:, 1:3].values
    # merge train and validation dataset
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    db_all = dataset_total.iloc[:, 1:3].values

    inputs = db_all[len(dataset_total) - len(dataset_test) - PastnumDays:]

    inputs = inputs.reshape(-1,2)

    inputs = sc.transform(inputs)
    X_test = []
    for i in range(PastnumDays, len(inputs)):
        X_test.append(inputs[i-PastnumDays:i, :])
        
    X_test = np.array(X_test)
    print(X_test.shape)
    print(X_test.shape[0])
    print(X_test.shape[1])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))
    
    return X_test, real_gold_price

#%%
# Create lstm model
def gold_model(X_train, y_train):
    model = Sequential()
    # The input of LSTM is form of [samples, timesteps, features]
    model.add(LSTM(units = 50, return_sequences = True,
          input_shape = (X_train.shape[1], 2)))
    
    model.add(Dropout(0.2))
    
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    # Fully connected NN - 2 outputs
    model.add(Dense(units = 2))
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    history = model.fit(X_train, y_train, epochs = 50, batch_size = 32)
    
    return model, history

#%%
# Train the model
# gr = [(i+1)*5 for i in range(36)]
# mse_all = []
# for past_days in gr:
past_days = 95
X_train, y_train, dataset_train, sc = train_data(past_days, train)
model, history = gold_model(X_train, y_train)
X_test, real_gold_price = test_data(past_days, dataset_train, test, sc)
predicted_gold_price = model.predict(X_test)
predicted_gold_price = sc.inverse_transform(predicted_gold_price)
  # mse = np.mean((predicted_gold_price - real_gold_price)**2)
  # mse_all.append((past_days,mse))

# %%
plt.plot(history.history['loss'])


plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# %%
print(np.mean(predicted_gold_price - real_gold_price) ** 2)

# %%
plt.plot(real_gold_price, color = 'black', label = 'Gold Price')
plt.plot(predicted_gold_price[:,0], color = 'green', label = 'Predicted Gold Price')
plt.title('Gold Price Prediction')
plt.xlabel('Time')
plt.ylabel('Gold Price')
plt.legend()
plt.show()


# %%
# predict for whole dataset
X_test, real_gold_price = test_data(90, dataset_train, df, sc)
predicted_gold_price = model.predict(X_test)
predicted_gold_price = sc.inverse_transform(predicted_gold_price)

# %%
print(np.mean(predicted_gold_price - real_gold_price) ** 2)

# %%
plt.plot(real_gold_price, color = 'black', label = 'Gold Price')
plt.plot(predicted_gold_price[:,0], color = 'green', label = 'Predicted Gold Price')
plt.title('Gold Price Prediction')
plt.xlabel('Time')
plt.ylabel('Gold Price')
plt.legend()
plt.show()


# %%
# Create lstm model
def gold_GRU_model(X_train, y_train):
    model = Sequential()
    # The input of GRU is form of [samples, timesteps, features]
    model.add(GRU(units = 50, return_sequences = True,
          input_shape = (X_train.shape[1], 2)))
    
    model.add(Dropout(0.2))
    
    model.add(GRU(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    
    model.add(GRU(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    
    model.add(GRU(units = 50))
    model.add(Dropout(0.2))
    # Fully connected NN - 2 outputs
    model.add(Dense(units = 2))
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    history = model.fit(X_train, y_train, epochs = 50, batch_size = 32)
    
    return model, history

# %%
# Train the model
# gr = [(i+1)*5 for i in range(15,26)]
# mse_all = []
# for past_days in gr:
X_train, y_train, dataset_train, sc = train_data(past_days, train)
model, history = gold_GRU_model(X_train, y_train)
X_test, real_gold_price = test_data(past_days, dataset_train, test, sc)
predicted_gold_price = model.predict(X_test)
predicted_gold_price = sc.inverse_transform(predicted_gold_price)
# mse = np.mean((predicted_gold_price - real_gold_price)**2)
# mse_all.append((past_days,mse))

# # %%
# grP = pd.DataFrame({'past_day':[i[0] for i in mse_all], 'mse': [i[1] for i in mse_all]})

# # %%
# grP.sort_values('mse')

# %%
# # Train the model
# X_train, y_train, dataset_train, sc = train_data(past_days, train)

# model = gold_GRU_model(X_train, y_train)

# %%
X_test, real_gold_price = test_data(past_days, dataset_train, test, sc)
predicted_gold_price = model.predict(X_test)
predicted_gold_price = sc.inverse_transform(predicted_gold_price)
print(np.mean((predicted_gold_price - real_gold_price)**2))
plt.plot(real_gold_price, color = 'black', label = 'Gold Price')
plt.plot(predicted_gold_price[:,0], color = 'green', label = 'Predicted Gold Price')
plt.title('Gold Price Prediction')
plt.xlabel('Time')
plt.ylabel('Gold Price')
plt.legend()
plt.show()

# %%
X_test, real_gold_price = test_data(past_days, dataset_train, df, sc)
predicted_gold_price = model.predict(X_test)
predicted_gold_price = sc.inverse_transform(predicted_gold_price)
print(np.mean((predicted_gold_price - real_gold_price)**2))
plt.plot(real_gold_price, color = 'black', label = 'Gold Price')
plt.plot(predicted_gold_price[:,0], color = 'green', label = 'Predicted Gold Price')
plt.title('Gold Price Prediction')
plt.xlabel('Time')
plt.ylabel('Gold Price')
plt.legend()
plt.show()
