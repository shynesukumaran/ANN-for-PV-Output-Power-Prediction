# ANN-for-PV-Output-Power-Prediction
Design an ANN model to predict the future output power of a PV(Photovoltaic) power plant.

The main aim is to use the historical data collected from a PV power plant for predicting the future output power values, preferably for next 24 or 38 hours.

The historical data collected from the power plant contains the output power values for the past 2 years and the data of some weather parameters measured at the site of the power plant. An ANN(Artificial Neural Network) model should be designed which takes the weather parameters as the input and predicts the PV output power. 

The historical weather parameters considered/collected for the project are: Wind speed, Air temperature and Solar Radiation(GHI).

Initially the ANN model is trained with the historical data of weather parameters and the PV output power. After the model is trained, weather forecast values are fed into the ANN model for the next 24 or 48 hours with the help of an API call to a weather website and the PV output power predictions are achieved.
