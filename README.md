# Forecasting Electric Load based on Weather Data

## Introduction
In this project, weather patterns data from various cities in Spain are explored and used to develop a supervised models to predict electric loads for these particular regions. That is a [linear regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), a support vector regression model [(SVR)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html), two [fully connected neural networks](https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html), and Long-short term memory (LSTM) networks were developed and trained with the weather patterns data (see [Pytorch on LSTM] (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) documentation for more details). Specifically, the linear regression model was trained with an L2 regularization parameter, the support vector regression model was trained with a radial basis function (RBF) kernel, and the fully connected neural network models had one and five hidden layers, respectively with a Tanh activation function, and were trained for 500 and 1000 epochs, respectively. Results from these models showed that more sophisticated models would be required to gain more insight from the data, therefore, LSTM models were developed and trained for 10 epochs. Also, note that the model evaluation metric throughout this project is the mean squared error.

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

To obtain a data format that was suitable for training the models, a series of data transformation steps were applied to each dataset. First, data points in the weather pattern dataset were records of each city and for every hour from 01-01-2015 to 12-31-2018, therefore, the data
was first split into records for each individual city and then aggregated to find the mean measurements for all five cities. In addition, missing values were filled in with linear
interpolation. Next, data points in the electric load dataset contained some missing values, thus, the linear interpolation method was used to fill in these missing values. Finally, the two datasets were intersected into a single dataset based on their timestamp columns. The final dataset contains 38568 sample points of which 30855 sample points (80%) were used for training and 7713 sample points (20%) were used for validation.

Moving on to the models and parameter tuning. First, a linear regression model with an L2 regularization parameter was trained. This is the ridge regression model from the Sckit-learn machine learning library. Using [grid search cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) tools from Sckit-learn, the model's best value for the alpha hyper-parameter was obtained as 1.0. Second, a support vector regression model with an RBF kernel was trained and using parameter tuning, the best values for gamma, and C hyper-parameters were reported as 10 and 1000, respectively. Third, two fully connected neural networks were developed and trained for 500 and 1000 epochs, respectively. The first neural network was tuned with 3 hidden layers and 280 nodes and trained with a stochastic gradient descent [(SGD) optimizer](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html), and the second neural network was configured with 5 hidden layers and 681 nodes, and also trained with an SDG optimizer. Finally, an LSTM network was developed and trained for 10 epochs. First, the LSTM network was configured with 1 layer and 64 hidden units and trained with an [Adam optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html). And finally, the network was re-configured with 3 layers, 64 hidden units, and retrained with the Adam optimizer.

## Results and Analysis

First, the linear regression model reported a mean squared error of 0.0356 and the training time was 0.409 seconds. Now, these results suggest a fair performance of the linear model, however, keep in mind that this model did not take into account the time component of the data, which leads to conclude that these results, though they might be promising, do not capture the full aspect of the data. Second, the support vector regression (SVR) reported a mean squared error of 0.0340 and completed training in 16397.253 seconds which converts to approximately 4.5 hours. Similarly, the SVR model presents promising results, however, they are not representative of the complete dataset. Next, the first fully connected neural network reported a training loss of 0.8969 and a validation loss of 0.8176 after 500 epochs, and the second neural network reported a training loss of 0.8919 and a validation loss of 0.8158 after 1000 epochs. Both networks had suboptimal results compared to the SVR and linear regression models. Figure 3(a) shows the loss curves of the first fully connected neural network and figure 3(b) shows the
loss curve of the second fully connected neural network.

![Figure 3(a)](https://github.com/claudeshyaka/ml-final-project/blob/main/images/loss_fun_3.png?raw=true) "Figure 3(a): Loss curve of the fully connected network with 3 hidden layers and 280 nodes."

![Figure 3(b)](https://github.com/claudeshyaka/ml-final-project/blob/main/images/loss_fun_5.png?raw=true) "Figure 3(b): Loss curve of the fully connected network with 5 hidden layers and 681 nodes."

Finally, the LSTM network configured with 1 layer, 64 hidden units, and an Adam optimizer reported a training loss of 0.4639 and a validation loss of 0.5457 after 10 epochs, and the LSTM network configured with a 3 layer, 64 hidden units, and the Adam optimizer reported a training loss of 0.4636 and a validation loss of 0.5339. Figure 4 (a) shows a plot of the
predicted and actual electric load values over time for the LSTM trained with the Adam optimizer and 1 layer, and Figure 4 (b) shows a plot of the LSTM trained with the Adam optimizer and 3 layers, in addition, for both figures the black dotted line indicates the beginning of the validation data points.

![Figure 4(a)](https://github.com/claudeshyaka/ml-final-project/blob/main/images/prediction_LSTM_1_layer.png?raw=true) "Figure 4(a): Predicted and actual electric load values over time for the LSTM trained with the
Adam optimizer and 1 layer."

![Figure 4(b)](https://github.com/claudeshyaka/ml-final-project/blob/main/images/prediction_LSTM_3_layers.png?raw=true) "Figure 4(b): Predicted and actual electric load values over time for the LSTM with 3 layers.
"