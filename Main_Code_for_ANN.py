#Load all the necessary libraries

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

#Load historical data from the local storage

path = r'*YOUR PATH TO THE FILE*'
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df) 
frame = pd.concat(li, axis=1, ignore_index=True)
print(frame)

# Selecting required columns and dropping repeating columns

frame =frame.rename(columns={0 : 'Date and Time',1:'Active Power',2:'D1',3:'Air Temperature',4:'D2',5:'Radiation',6:'D3',7:'Wind Speed'})
finaldata = frame.drop(columns=['D1', 'D2','D3'])
finaldata.set_index('Date and Time')

#Plotting and checking the input data

import matplotlib.dates as mdates
ax = finaldata.plot(x = 'Date and Time',y = 'Active Power')
plt.gcf().autofmt_xdate()
plt.show()

#Training ANN model
#Splitting dataset into training and test set

train, test = train_test_split(finaldata, train_size=0.80, shuffle=False)

#Normalizing data

cols_to_norm = ['Active Power','Air Temperature', 'Radiation','Wind Speed']
train[cols_to_norm] = train[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
test[cols_to_norm] = test[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(train)
print(test)

train.set_index(['Date and Time'])
test.set_index(['Date and Time'])
X_train = train.iloc[:,2:5]
y_train = train['Active Power'].values
X_test = test.iloc[:,2:5]
y_test = test['Active Power'].values
print(X_train)
print(y_train)
print(X_test)
print(y_test)

#Defining ANN model

nn_model = Sequential()
nn_model.add(Dense(64, input_dim=3, activation = 'relu'))
nn_model.add(Dense(16, activation = 'relu'))
nn_model.add(Dense(8, activation = 'relu'))
nn_model.add(Dense(1))
nn_model.summary()

#Training the model

nn_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history = nn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, callbacks=[early_stop], shuffle=False)

#Saving the trained model

import pickle
with open('model_pickle','wb') as g:
    pickle.dump(nn_model,g)

#Making predictions  
    
y_pred_test_nn = nn_model.predict(X_test)
y_train_pred_nn = nn_model.predict(X_train)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_nn)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_nn)))

#Combining predicted and actual data for plotting

date = test['Date and Time'].values
predicted_testdata = y_pred_test_nn.flatten()
final_results_testdata = pd.DataFrame({"Date and Time": date, "Actual Active Power": y_test, "Predicted Active Power": predicted_testdata})
print(final_results_testdata)

#Plotting results for comparison

fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(20,10),sharex=True, sharey=True)
ax = plt.gca()
fig.suptitle('Comparison: Actual vs Predicted(For Test data from the Power Plant)', fontsize=20)
final_results_testdata.plot(kind='line',x='Date and Time',y='Actual Active Power',color = '#33D1FF', ax=axes[0])
final_results_testdata.plot(kind='line',x='Date and Time',y='Predicted Active Power', color='#FF3333', ax=axes[1])
plt.xlabel('Date and Time', fontsize=16)
fig.text(0.04,0.6,'Active Power in Watts (normalised values)', va='center', rotation='vertical',fontsize=16)
plt.gcf().autofmt_xdate()
plt.savefig('Test_data_prediction.png', dpi = 300)
plt.show()


#Weather forecast data archive from a weather website(SOLCAST) is used for the same period of test data for another comparison

#Loading the SOLCAST data from the loacl storage
solcast_test_data = pd.read_csv(r"*YOUR PATH TO THE FILE*", sep = ",",decimal='.')
print (solcast_test_data)

#Normalizing the data and selecting the input columns

columns_to_normalize = ['Air Temperature', 'Radiation','Wind Speed']
solcast_test_data[columns_to_normalize] = solcast_test_data[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(solcast_test_data)
solcast_test_data.set_index(['Date and Time'])
X_solcast_testdata = solcast_test_data.iloc[:,1:5]
print(X_solcast_testdata)

#Predicting using the trained ANN model

y_pred_solcast_testdata_nn = nn_model.predict(X_solcast_testdata)
print(y_pred_solcast_testdata_nn)

#Combining predictions with actual data for comparison

predicted_solcast_testdata = y_pred_solcast_testdata_nn.flatten()
print(predicted_solcast_testdata)
final_results_solcast_testdata = pd.DataFrame({"Date and Time": date, "Actual Active Power": y_test, "Predicted Active Power": predicted_solcast_testdata})
print(final_results_solcast_testdata)

#Plotting results for comparison

fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(20,10),sharex=True, sharey=True)
ax = plt.gca()
fig.suptitle('Comparison: Actual vs Predicted(Using SOLCAST data as Test set)', fontsize=20)
final_results_solcast_testdata.plot(kind='line',x='Date and Time',y='Actual Active Power',color = '#33D1FF', ax=axes[0])
final_results_solcast_testdata.plot(kind='line',x='Date and Time',y='Predicted Active Power', color='#FF3333', ax=axes[1])
plt.xlabel('Date and Time', fontsize=16)
fig.text(0.04,0.6,'Active Power in Watts (normalised values)', va='center', rotation='vertical',fontsize=16)
plt.gcf().autofmt_xdate()
plt.savefig('Test_data_prediction(Active Power).png', dpi = 300)
plt.show()


