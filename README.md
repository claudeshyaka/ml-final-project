# Forecasting Electric Load based on Weather Data

## Introduction
In this project, weather patterns data from various cities in Spain are explored and used to develop a supervised models to predict electric loads for these particular regions. That is a [linear regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), a support vector regression model ([SVR])(https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html), two [fully connected neural networks](https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html), and Long-short term memory (LSTM) networks were developed and trained with the weather patterns data (see Pytorch on LSTM documentation for more details)[LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html). Specifically, the linear regression model was trained with an L2 regularization parameter, the support vector regression model was trained with a radial basis function (RBF) kernel, and the fully connected neural network models had one and five hidden layers, respectively with a Tanh activation function, and were trained for 500 and 1000 epochs, respectively. Results from these models showed that more sophisticated models would be required to gain more insight from the data, therefore, LSTM models were developed and trained for 10 epochs. Also, note that the model evaluation metric throughout this project is the mean squared error.

## Data
Data used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather?select=weather_features.csv). As mentioned in the introduction,
the weather patterns data used in this project is for five different cities in Spain (Madrid,
Barcelona, Bilbao, Valencia, and Seville), and the electric load data is for all five cities
combined. In addition, the weather patterns dataset contained the following 16 columns and
178396 instances and the electric load dataset contained 35064 instances and 28 columns. Please
see the [data-processing](https://github.com/claudeshyaka/ml-final-project/blob/main/data_processing.ipynb) Jupyter notebook in the Github repo for more details. After a preview
of the datasets, the following filter was applied on the weather patterns dataset: time, city_name,
temperature, pressure, humidity, rain_1h, rainh_3h, and swon_3h; and for the electric load
dataset, the following filter was used: time, total load actual.