# Nonlinear-Factor-Model

The module nDFM_module implements the simulation and estimation algorithms presented in the paper "Nonlinear Dynamic Factor Modeling and Estimation via Autoencoding Neural Networks", soon available on arXiv. The link will be added here.

The file example.py shows how to use the module to simulate data from a nonlinear dynamic factor model, make oracle predictions, estimate linear and nonlinear dynamic factor models, and compare their performance.

nDFM_module contains the following classes:
- autoencoder : trains an autoencoder and makes predictions
- TimeSeriesMLP : trains a MLP to make predictions based on lagged observations
- diagonal_VAR : estimates a vector autoregressive model with diagonal autoregressive matrices
- nDFM : estimates a nonlinear dynamic factor model
- VAR : estimates a vector autoregressive model
- DFM : estimates a linear dynamic factor model via PCA
- nDFM_simulator : simulates data from a nonlinear dynamic factor model and makes oracle predictions


autoencoder: Autoencoder class. Inherits from the keras.models.Sequential class. Can be used to train an autoencoder, encode data and decode previously encoded data. Available methods:
- train: Trains an autoencoder with specified architecture and given data.
- encode: Encodes data based on the trained network.
- decode: Decodes encoded data based on the trained network.


TimeSeriesMLP:

Class to train and forecast multiple time series using a multilayer 
perceptron. 

Available methods:
- train: Trains the network for a given number of lags.
- train_CV: Trains the network for a list of numbers of lags and chooses the best-performing number of lags on a validation set.
- forecast: computes 1-step-ahead forecasts based on the trained network.

The method forecast can only be used after running the method train or train_CV.


diagonal_VAR:

Class to train a vector autogregressive (VAR) model with diagonal coefficient matrices. I.e., equivalent to training autoregressive models for each of the provided time series individually.

Available methods:
- train: Trains a diagonal VAR for a given number of lags.
- forecast: Computes 1-step-ahead forecasts based on the trained VAR

The method forecast can be used only after running the method train.


nDFM:

Class to estimate a nonlinear dynamic factor model via an autoencoder and make predictions.

Available methods:
- train: train the nonlinear factor model for given factor lags, idiosyncratic lags, factors and neurons in the hidden layers.
- train_CV: train the nonlinear factor model via a grid search for the hyperparameters, choosing the model minimizing the validation MSE.
- forecast: computes 1-step-ahead forecasts based on the estimated model.

The method forecast can be used only after having estimated the model, i.e. after having run the method train or the method train_CV.


VAR:

Class to estimate a vector autoregressive (VAR) model and make predictions.

Available methods:
- train: Estimates the model for a given number of lags.
- train_CV: Estimates the model for a list of numbers of lags and chooses the best-performing number of lags on a validation set.
- forecast: computes 1-step-ahead forecasts based on the estimated model.

The method forecast can only be used after running the method train or train_CV.

 

DFM:

Class to estimate a linear dynamic factor model via principal component analysis (PCA) and make predictions.

Available methods:
- train: train the linear factor model for given factor lags, idiosyncratic lags and factors.
- train_CV: train the linear factor model via a grid search for the hyperparameters, choosing the model minimizing the validation MSE.
- forecast: computes 1-step-ahead forecasts based on the estimated model.

The method forecast can be used only after having estimated the model, i.e.
after having run the method train or the method train_CV.


nDFM_simulator:

Class to initialize a nonlinear dynamic factor model (nDFM), simulate data and make oracle forecasts.

Available methods:
- simulate: Simulate data from a nDFM with specified structure
- predict_oracle: make oracle 1-step-ahead forecasts for the simulated data

The method predict_oracle can be used only after having simulated data via the method simulate.
