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
df = pd.read_csv('./data/LBMA-GOLD.csv')
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

# Get training data
def train_data(PastnumDays, dataset_train):
    training_set = dataset_train.iloc[:, 1:2].values
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

# Get test data
def test_data(PastnumDays, dataset_train, dataset_test, sc):
    real_stock_price = dataset_test.iloc[:, 1:2].values
    # merge train and validation dataset
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    db_all = dataset_total.iloc[:, 1:2].values

    inputs = db_all[len(dataset_total) - len(dataset_test) - PastnumDays:]
    inputs = inputs.reshape(-1,2)

    inputs = sc.transform(inputs)
    X_test = []
    for i in range(PastnumDays, len(db_all)):
        X_test.append(inputs[i-PastnumDays:i, :])
        #X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))
    
    return X_test,real_stock_price

# Create lstm model
def stock_model(X_train, y_train):
    model = Sequential()
    # The input of LSTM is form of [samples, timesteps, features]
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 2)))
    
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
    model.fit(X_train, y_train, epochs = 100, batch_size = 32)
    
    return model

def main():
    X_train, y_train,dataset_train,sc = train_data()
    
    model = stock_model(X_train, y_train)
    X_test,real_stock_price = test_data(dataset_train,sc)
    predicted_stock_price = model.predict(X_test)
    
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    plt.plot(real_stock_price, color = 'black', label = 'Gold Price')
    
    plt.plot(predicted_stock_price[:,0], color = 'green', label = 'Predicted Gold Price')
    plt.title('Gold Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
