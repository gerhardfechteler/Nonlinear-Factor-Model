import numpy as np
import keras
from keras.backend import clear_session
import statsmodels.api as sm
from numpy.linalg import inv



class autoencoder(keras.models.Sequential):
    """
    Autoencoder class. Inherits from the keras.models.Sequential class. Can be
    used to train an autoencoder, encode data and decode previously encoded 
    data. 
    
    Available methods:
    - train: Trains an autoencoder with specified architecture and given data.
    - encode: Encodes data based on the trained network.
    - decode: Decodes encoded data based on the trained network.
    
    The methods encode and decode can be used only after training.
    """
    
    def train(self, X_raw, number_of_factors, w=100):
        """
        train(self, X_raw, number_of_factors, w=100)
        
        Trains an outoencoder with 5 layers. The width of input and output 
        layer is given by the dimension of the data, the width of second and 
        fourth layer by w, and the width of the third layer by 
        number_of_factors.

        Parameters
        ----------
        X_raw: numpy.ndarray
            (n, k) array of observed data
            n - number of observations
            k - number of variables
        number_of_factors: int
            number of neurons in the bottleneck layer / factors to estimate
        w : int, optional
            Number of neurons in the second and fourth layer, respectively. 
            The default is 100.

        Returns
        -------
        None.
        """
        
        # obtain number of observations and series
        n, k = X_raw.shape
        
        # standardize input
        self.X_mean = np.mean(X_raw, 0)
        self.X_std = np.std(X_raw, 0)
        X = (X_raw - self.X_mean) / self.X_std
        
        # Model Architecture
        self.add(keras.layers.Dense(w, 
                                    activation = 'tanh',
                                    input_dim = k,
                                    kernel_regularizer = keras.regularizers.l2(0.001)))
        self.add(keras.layers.Dense(number_of_factors))
        self.add(keras.layers.Dense(w,
                                    activation = 'tanh',
                                    kernel_regularizer=keras.regularizers.l2(0.001)))
        self.add(keras.layers.Dense(k))
        
        # Compilation
        self.compile(optimizer = keras.optimizers.Adam(lr=0.015, 
                                                       beta_1=0.9, 
                                                       beta_2=0.999, 
                                                       epsilon=1E-8,
                                                       amsgrad=False),
                     loss = 'mse')
        
        # Training
        self.fit(X, X,
                 epochs = 2000,
                 batch_size = 128,
                 validation_split = 0.1,
                 callbacks = [keras.callbacks.EarlyStopping(patience = 500,
                                                            restore_best_weights = True)],
                 verbose = 0)
    
    
    def encode(self, X_raw):
        """
        encode(self, X_raw)
        
        Encodes the provided data and returns the estimated corresponding 
        factors.

        Parameters
        ----------
        X_raw : numpy.ndarray
            (m,k) array of observed data.
            m - number of observations to decode
            k - number of variables, has to match training data.

        Returns
        -------
        F : numpy.ndarray
            (m,number_of_factors) array of encoded data.
            m - number of observations to decode
            number_of_factors - as provided in the method train.
        """
        
        # standardize series
        X = (X_raw - self.X_mean) / self.X_std
        
        # obtain encoder
        encoder = keras.models.Sequential()
        for layer in self.layers[:2]:
            encoder.add(layer)
        
        # obtain encoded series
        F = encoder.predict(X)
        
        return F
    
    
    def decode(self, F):
        """
        decode(self, F)
        
        Decode the provided factors/decoded data and return the reconstructed
        data.

        Parameters
        ----------
        F : numpy.ndarray
            (m,number_of_factors) array of encoded data.
            m - number of observations to decode
            number_of_factors - as provided in the method train.

        Returns
        -------
        X_decoded : numpy.ndarray
            (m,k) array of decoded data.
            m - number of observations to decode
            k - number of variables, has to match training data.
        """
        
        # obtain decoder
        decoder = keras.models.Sequential()
        for layer in self.layers[2:]:
            decoder.add(layer)
        
        # obtain rescaled predicted series
        X_decoded = decoder.predict(F) * self.X_std + self.X_mean
        
        return X_decoded



class TimeSeriesMLP():
    """
    Class to train and forecast multiple time series using a multilayer 
    perceptron. 
    
    Available methods:
    train: Trains the network for a given number of lags.
    train_CV: Trains the network for a list of numbers of lags and chooses the
        best-performing number of lags on a validation set.
    forecast: computes 1-step-ahead forecasts based on the trained network.
    
    The method forecast can only be used after running the method train or
    train_CV.
    """
    
    def __init__(self):
        pass
    
    
    def train(self, X_raw, lags=1, w=100):
        """
        train(self, X_raw, lags=1, w=100)
        
        Trains a multilayer perceptron with one hidden layer, using the lagged
        values of the series as regressors.

        Parameters
        ----------
        X_raw : numpy.ndarray
            (T,k) array of observed series
            T - number of observations
            k - number of series
        lags : int, optional
            Number of lags. The default is 1.
        w : int, optional
            Width of the hidden layer. The default is 100.

        Returns
        -------
        None.
        """
        
        # initialize network
        self.MLP = keras.models.Sequential()
        
        # numer of observations, number of series and lags
        T, k = X_raw.shape
        self.lags = lags
        
        # Standardize series
        self.X_mean = np.mean(X_raw, 0)
        self.X_std = np.std(X_raw, 0)
        X = (X_raw - self.X_mean) / self.X_std
        
        # Model architecture
        self.MLP.add(keras.layers.Dense(w, 
                                        activation = 'tanh',
                                        input_dim = k * lags,
                                        kernel_regularizer = keras.regularizers.l2(0.001)))
        self.MLP.add(keras.layers.Dense(k))
        
        # Compilation
        self.MLP.compile(optimizer = keras.optimizers.Adam(lr=0.015, 
                                                           beta_1=0.9, 
                                                           beta_2=0.999, 
                                                           epsilon=1E-8,
                                                           amsgrad=False),
                     loss = 'mse')
        
        # Prepare Data
        X_x = np.hstack(tuple(X[i:T-lags+i,:] for i in range(lags)))
        X_y = X[lags:,:]
        
        # Training
        self.MLP.fit(X_x, X_y,
                     epochs = 300,
                     batch_size = 512,
                     validation_split = 0.1,
                     callbacks = [keras.callbacks.EarlyStopping(patience = 50,
                                                                restore_best_weights = True)],
                     verbose = 0)
    
    
    def train_CV(self, X, range_lags = [i+1 for i in range(4)], w=100):
        """
        train_CV(self, X, range_lags = [i+1 for i in range(4)], w=100)
        
        Trains a MLP for the series X. Chooses the number of lags in the given 
        range that minimizes the validation MSE. The last 10% of the 
        observations are used for validation. The best model is trained again 
        on the whole set.

        Parameters
        ----------
        X : numpy.ndarray
            (T,k) array of observed series
            T - number of observations
            k - number of series
        range_lags: list, optional
            list of integer numbers of lags to consider. 
            The default is [1,2,3,4].
        w : int, optional
            Width of the hidden layer. The default is 100.

        Returns
        -------
        None.
        """
        
        T,k = X.shape
        
        # split the data in training and validation set
        split = int(np.round(0.9*T)) # index for splitting in training and validation
        X_train = X[:split, :] # training set
        X_val = X[split:, :] # validation set
        
        MSE = np.zeros(len(range_lags))
        for i,lags in enumerate(range_lags):
            # train the network
            clear_session()
            self.train(X_train, lags=lags, w=w)
            
            # compute the validation MSE
            X_pred = self.forecast(X_val)
            MSE[i] = np.mean((X_val[lags:, :] - X_pred[:-1, :])**2)
            
        # obtain MSE optimal hyperparameters
        self.best_lags = range_lags[np.argmin(MSE)]
        print(MSE)
        
        # train the model for the number of lags minimizing the cross-val MSE
        self.train(X_train, self.best_lags)
    
    
    def forecast(self, X_raw):
        """
        forecast(self, X_raw)
        
        Computes 1-step-ahead forecasts of the provided series.

        Parameters
        ----------
        X_raw : numpy.ndarray
            (m,k) array of observed series
            m - number of observations, has to be larger or equal than number 
                of lags
            k - number of series

        Returns
        -------
        X_forecast : numpy.ndarray
            (m-lags+1,k) array of forecasted series
            m - number of observations
            k - number of series
        """
        
        # obtain length and number of series
        m, k = X_raw.shape
        
        # Standardize series
        X = (X_raw - self.X_mean) / self.X_std
        
        # Prepare Data
        X_x = np.hstack(tuple(X[i:m-self.lags+i+1,:] for i in range(self.lags)))
        
        # predict output
        X_pred = self.MLP.predict(X_x)
        
        # obtain rescaled forecasted variables
        X_forecast = X_pred * self.X_std + self.X_mean
        
        return X_forecast



class diagonal_VAR():
    """
    Class to train a vector autogregressive (VAR) model with diagonal 
    coefficient matrices. I.e., equivalent to training autoregressive models 
    for each of the provided time series individually.
    
    Available methods:
    - train: Trains a diagonal VAR for a given number of lags.
    - forecast: Computes 1-step-ahead forecasts based on the trained VAR
    
    The method forecast can be used only after running the method train.
    """
    
    def __init__(self):
        pass
    
    
    def train(self, X, lags):
        """
        train(self, X, lags)
        
        Estimates the autoregressive parameters (diagonal matrices) of the VAR
        model for the provided number of lags.

        Parameters
        ----------
        X : numpy.ndarray
            (T,k) array of observed series.
        lags : int
            Number of lags.

        Returns
        -------
        None.
        """
        
        T,k = X.shape
        beta = np.zeros((k,lags))
        
        # as the VAR model is diagonal, we can estimate each series separately
        for i in range(k): # iteration over all series
            # Construct data, lagged values are regressors
            Yi = X[lags:,i]
            Xi = np.zeros((T-lags, lags))
            for lag in range(lags):
                Xi[:,lags-lag-1] = X[lag:T-lags+lag,i]
            
            # Estimate the parameters by OLS
            beta[i,:] = (inv(Xi.T @ Xi) @ Xi.T @ Yi).flatten()
        
        self.lags = lags
        self.k = k
        self.beta = beta
    
    
    def forecast(self, X):
        """
        forecast(self, X)
        
        Computes 1-step-ahead forecasts for the provided series based on the
        trained model.

        Parameters
        ----------
        X : numpy.ndarray
            (m,k) array of observed series
            m - number of observations, has to be larger or equal than number 
                of lags
            k - number of series

        Returns
        -------
        X_pred : numpy.ndarray
            (m-lags+1,k) array of forecasted series
            m - number of observations
            k - number of series
        """
        
        T,k = X.shape
        lags = self.lags
        
        # Check for consistency between training and forecasting data
        if k != self.k:
            raise ValueError('The number of series differs between training and forecast data.')
        
        X_pred = np.zeros((T-lags+1, k))
        
        # we forecast each series separately, as the VAR is diagonal
        for i in range(k):
            Xi = np.zeros((T-lags+1, lags))
            for lag in range(lags):
                Xi[:,lags-lag-1] = X[lag:T-lags+lag+1,i]
            X_pred[:,i] = (Xi @ self.beta[i,:].reshape((-1,1))).flatten()
        
        return X_pred



class nDFM():
    """
    Class to estimate a nonlinear dynamic factor model via an autoencoder and
    make predictions.
    
    Available methods:
    - train: train the nonlinear factor model for given factor lags, 
        idiosyncratic lags, factors and neurons in the hidden layers.
    - train_CV: train the nonlinear factor model via a grid search for the
        hyperparameters, choosing the model minimizing the validation MSE.
    - forecast: computes 1-step-ahead forecasts based on the estimated model.
    
    The method forecast can be used only after having estimated the model, i.e.
    after having run the method train or the method train_CV.
    """
    
    def __init__(self):
        pass
    
    def train(self, 
              X, 
              lags_factor_dynamics = 1, 
              lags_idiosyncratic_dynamics = 1, 
              number_of_factors = 2,
              training_method = 'state_space',
              verbose = True):
        """
        train(self, X, 
                  lags_factor_dynamics = 1, 
                  lags_idiosyncratic_dynamics = 1, 
                  number_of_factors = 2,
                  training_method = 'state_space',
                  verbose = True)
        
        Trains a nonlinear dynamic factor model for the series X.

        Parameters
        ----------
        X : numpy.ndarray
            (T,k) array of observed series
            T - number of observations
            k - number of series
        lags_factor_dynamics : int, optional
            Number of lags to use for the estimation of the factor dynamics. 
            The default is 1.
        lags_idiosyncratic_dynamics : int, optional
            Number of lags to use for the estimation of the idiosyncratic noise
            dynamics. The default is 1.
        number_of_factors : int, optional
            Number of factors to estimate, i.e. number of neurons in the 
            bottleneck layer. The default is 2.
        training_method : str, optional
            Method used for training, should be one of:
            - 'state_space': uses the state space representation for estimation
            - 'VAR': uses the VAR representation for estimation
            The default is 'state_space'.
        verbose : bool, optional
            Indicates, whether updates on the training progress should be 
            outputted in the console. The default is True.

        Returns
        -------
        None.
        """
       
        self.verbose = verbose
        
        # Clear the keras session and initialize the model components
        clear_session()
        self.autoencoder = autoencoder()
        self.factor_dynamics = TimeSeriesMLP()
        self.noise_dynamics = diagonal_VAR()
        
        # save global variables
        self.training_method = training_method
        self.lags_factor_dynamics = lags_factor_dynamics
        self.lags_idiosyncratic_dynamics = lags_idiosyncratic_dynamics
        self.number_of_factors = number_of_factors
        
        # call training methods based on training_method
        if (training_method == 'state_space'):
            self.__train_state_space__(X)
        elif (training_method == 'VAR'): # VAR estimation not implemented
            # self.__train_VAR__(X)
            raise ValueError('VAR estimation method not yet implemented.')
        else:
            raise ValueError('The training method should be state_space of VAR')
    
    def train_CV(self, X, 
                  range_lags_fd = [i+1 for i in range(5)], 
                  range_lags_id = [i+1 for i in range(3)], 
                  range_numfac = [i+1 for i in range(5)],
                  training_method = 'state_space',
                  verbose = True):
        """
        train_CV(self, X, 
                      range_lags_fd = [i+1 for i in range(5)], 
                      range_lags_id = [i+1 for i in range(3)], 
                      range_numfac = [i+1 for i in range(5)],
                      training_method = 'state_space',
                      verbose = True)
        
        Trains a nDFM for the series X. Chooses the hyperparameters in the 
        given ranges that minimize the validation MSE. The last 10% of 
        observations are used for validation. The best model is trained
        again on the whole set.

        Parameters
        ----------
        X : numpy.ndarray
            (T,k) array of observed series
            T - number of observations
            k - number of series
        range_lags_fd : list, optional
            List of integers, number of lags to consider for the factor 
            dynamics. The default is [1,2,3,4,5].
        range_lags_id : list, optional
            List of integers, numbers of lags to consider for the idiosyncratic 
            dynamics. The default is [1,2,3].
        range_numfac : list, optional
            List of integers, numbers of factors to consider. The default is 
            [1,2,3,4,5].
        training_method : str, optional
            Method used for training, should be one of:
            - 'state_space': uses the state space representation for estimation
            - 'VAR': uses the VAR representation for estimation
            The default is 'state_space'.
        verbose : bool, optional
            Indicates, whether updates on the training progress should be 
            outputted in the console. The default is True.

        Returns
        -------
        None.
        """
        
        self.verbose = verbose
        
        T,k = X.shape
        split = int(np.round(0.9*T)) # index for splitting in training and validation
        X_train = X[:split, :] # training set
        X_val = X[split:, :] # validation set
        MSE = np.zeros((len(range_numfac), len(range_lags_id), len(range_lags_fd)))
        for i,num_fac in enumerate(range_numfac):
            for j,lags_id in enumerate(range_lags_id):
                for k,lags_fd in enumerate(range_lags_fd):
                    if self.verbose == True:
                        print('Factor '+str(i+1)+' of '+str(len(range_numfac))+
                              ', idiosyncratic lag '+str(j+1)+' of '+str(len(range_lags_id))+
                              ', factor dynamic lag '+str(k+1)+' of '+str(len(range_lags_fd)))
                    maxlags = max([lags_fd, lags_id])
                    self.train(X_train, lags_fd, lags_id, num_fac, training_method, verbose)
                    X_pred = self.forecast(X_val)[1]
                    MSE[i,j,k] = np.mean((X_val[maxlags:, :] - X_pred[:-1, :])**2)
                    
        # obtain MSE optimal hyperparameters
        best_index = [x[0] for x in np.where(MSE == np.min(MSE))]
        self.best_numfac = range_numfac[best_index[0]]
        self.best_lags_id = range_lags_id[best_index[1]]
        self.best_lags_fd = range_lags_fd[best_index[2]]
        if self.verbose == True:
            print(MSE)
        
        # train the model for the number of lags minimizing the cross-val MSE
        self.train(X_train, 
                    self.best_lags_fd,
                    self.best_lags_id,
                    self.best_numfac,
                    training_method)
    
    def __train_state_space__(self, X):
        """
        __train_state_space__(self, X)
        
        Trains the model via the state space representation.

        Parameters
        ----------
        X : numpy.ndarray
            (T,k) array of observed series
            T - number of observations
            k - number of series

        Returns
        -------
        None.
        """
        
        # estimate autoencoder
        self.autoencoder.train(X, self.number_of_factors)
        
        # estimate factors
        F = self.autoencoder.encode(X)
        
        # estimate factor dynamics
        self.factor_dynamics.train(F, self.lags_factor_dynamics)
        
        # predict factors
        F_predicted = self.factor_dynamics.forecast(F)
        
        # obtain factor residuals
        self.F_resid = F[self.lags_factor_dynamics:, :] - F_predicted[:-1, :]
        
        if self.lags_idiosyncratic_dynamics > 0:
            # obtain idiosyncratic noise residuals
            X_resid = X - self.autoencoder.decode(F)
            
            # estimate idiosyncratic noise dynamics
            self.noise_dynamics.train(X_resid, self.lags_idiosyncratic_dynamics)
    
    def __train_var__(self, X):
        """
        __train_var__(self, X) 
        
        Trains the model via the VAR representation. Not implemented yet.

        Parameters
        ----------
        X : numpy.ndarray
            (T,k) array of observed series
            T - number of observations
            k - number of series

        Returns
        -------
        None.
        """
        
        pass
    
    
    def forecast(self, X):
        """
        forecast(self, X)
        
        Computes 1-step-ahead forecasts for the provided observations based on
        the estimated model.

        Parameters
        ----------
        X : numpy.ndarray
            (m,k) array of observed series
            m - number of observations, has to be larger or equal than number 
                of lags
            k - number of series

        Returns
        -------
        X_pred : numpy.ndarray
            (m-lags+1,k) array of forecasted series, not bias adjusted
            m - number of observations
            k - number of series
        X_pred_boot : numpy.ndarray
            (m-lags+1,k) array of forecasted series, bias adjusted
            m - number of observations
            k - number of series
        """
        
        try:
            if (self.training_method == 'state_space'):
                X_pred, X_pred_boot = self.__forecast_state_space__(X)
            elif (self.training_method == 'VAR'): # VAR not yet implemented
                # X_pred, X_pred_boot = self.__forecast_VAR__(X)
                raise ValueError('VAR training method not yet implemented.')
            else:
                raise ValueError('Training method unknown.')
        except:
            raise ValueError('Forecasting failed.')
        
        return X_pred, X_pred_boot 
    
    
    def __forecast_state_space__(self, X):
        """
        __forecast_state_space__(self, X)
        
        Computes 1-step-ahead forecasts for the provided observations based on
        the estimated state space representation of the model.

        Parameters
        ----------
        X : numpy.ndarray
            (m,k) array of observed series
            m - number of observations, has to be larger or equal than number 
                of lags
            k - number of series

        Returns
        -------
        X_pred : numpy.ndarray
            (m-lags+1,k) array of forecasted series, not bias adjusted
            m - number of observations
            k - number of series
        X_pred_boot : numpy.ndarray
            (m-lags+1,k) array of forecasted series, bias adjusted
            m - number of observations
            k - number of series
        """

        # estimate factors
        F = self.autoencoder.encode(X)
        
        # predict factors
        F_predicted = self.factor_dynamics.forecast(F)
        
        # predict series
        X_predicted = self.autoencoder.decode(F_predicted) #+X_resid_predicted
        
        # bootstrapped expectation
        X_predicted_boot = np.zeros(X_predicted.shape)
        n_boot = self.F_resid.shape[0] # number of available residuals
        batch = 10
        n_iter = int(np.ceil(n_boot/batch))
        for i in range(n_iter):
            resid = np.kron(self.F_resid[i*batch:(i+1)*batch,:], np.ones((X_predicted.shape[0],1)))
            batch_i = int(resid.shape[0] / X_predicted.shape[0])
            F_pred_batch = np.kron(np.ones((batch_i,1)), F_predicted)
            X_pred_i = self.autoencoder.decode(F_pred_batch + resid)
            for j in range(batch_i):
                X_predicted_boot += X_pred_i[j*X_predicted.shape[0]:(j+1)*X_predicted.shape[0],:]
        X_predicted_boot /= n_boot
        
        if self.lags_idiosyncratic_dynamics > 0:
            # obtain residuals to predict residual dynamics
            e = X - self.autoencoder.decode(F)
            e_predicted = self.noise_dynamics.forecast(e)
            
            # merge factor and noise components
            lags_diff = self.lags_idiosyncratic_dynamics - self.lags_factor_dynamics
            if lags_diff >= 0:
                X_predicted = X_predicted[-lags_diff:,:] + e_predicted
                X_predicted_boot = X_predicted_boot[lags_diff:,:] + e_predicted
            else:
                X_predicted += e_predicted[-lags_diff:,:]
                X_predicted_boot += e_predicted[-lags_diff:,:]
        
        return X_predicted, X_predicted_boot
    
    
    def __forecast_VAR__(self, X):
        """
        __forecast_state_space__(self, X)
        
        Computes 1-step-ahead forecasts for the provided observations based on
        the estimated state space representation of the model. Not implemented
        yet.

        Parameters
        ----------
        X : numpy.ndarray
            (m,k) array of observed series
            m - number of observations, has to be larger or equal than number 
                of lags
            k - number of series

        Returns
        -------
        None.
        """
        
        pass



class VAR():
    """
    Class to estimate a vector autoregressive (VAR) model and make predictions.
    
    Available methods:
    train: Estimates the model for a given number of lags.
    train_CV: Estimates the model for a list of numbers of lags and chooses the
        best-performing number of lags on a validation set.
    forecast: computes 1-step-ahead forecasts based on the estimated model.
    
    The method forecast can only be used after running the method train or
    train_CV.
    """
    
    def __init__(self):
        pass
    
    
    def train(self, X, lags=1):
        """
        train(self, X, lags=1)
        
        Estimates a VAR model with the given number of lags, using the lagged
        values of the series as regressors.

        Parameters
        ----------
        X : numpy.ndarray
            (T,k) array of observed series
            T - number of observations
            k - number of series
        lags : int, optional
            Number of lags. The default is 1.

        Returns
        -------
        None.
        """
        
        T, k = X.shape
        self.k = k
        self.lags = lags
        
        # Define the matrices corresponding to the matrix notation of the VAR
        Y1 = X[lags:,:].T
        Z1 = np.hstack(tuple([X[lag:T-lags+lag,:] for lag in range(lags)]))
        Z1 = sm.add_constant(Z1).T
        
        # Vectorize the matrix equation
        Y2 = Y1.reshape((-1,1))
        Z2 = np.kron(np.eye(k), Z1.T)
        
        # obtain the OLS parameter estimator
        A2 = inv(Z2.T @ Z2) @ Z2.T @ Y2
        
        # construct the parameter matrix
        A1 = A2.reshape((k, k*lags+1))
        self.A1 = A1
        
        # construct the autogregressive matrices
        A = [A1[:,0].reshape((-1,1))]
        for i in range(lags):
            A.append(A1[:,i*k+1:(i+1)*k+1])
        self.A = A
    
    
    def train_CV(self, X, range_lags = [i+1 for i in range(3)]):
        """
        Trains a VAR model for the series X. Chooses the number of lags in the given 
        range that minimizes the validation MSE. The last 10% of the 
        observations are used for validation. The best model is trained again 
        on the whole set.
        
        Parameters
        ----------
        X : numpy.ndarray
            (T,k) array of observed series
            T - number of observations
            k - number of series
        range_lags: list, optional
            list of integer numbers of lags to consider. 
            The default is [1,2,3].
        
        Returns
        -------
        None.
        """
        
        T,k = X.shape
        
        # split data in training and validation set
        split = int(np.round(0.9*T)) # index for splitting in training and validation
        X_train = X[:split, :] # training set
        X_val = X[split:, :] # validation set
        
        # estimate a VAR for each specified number of lags
        MSE = np.zeros(len(range_lags))
        for i,lags in enumerate(range_lags):
            print(str(i+1) + ' of ' + str(len(range_lags)))
            self.train(X_train, lags = lags)
            X_pred = self.forecast(X_val)
            MSE[i] = np.mean((X_val[lags:, :] - X_pred[:-1, :])**2)
        
        # identify the best model in terms of validation MSE
        self.best_lags = range_lags[np.argmin(MSE)]
        print(MSE)
        
        # train the model for the number of lags minimizing the validation MSE
        self.train(X, lags = self.best_lags)
    
    
    def forecast(self, X):
        """
        forecast(self, X)
        
        Computes 1-step-ahead forecasts of the provided series.

        Parameters
        ----------
        X : numpy.ndarray
            (m,k) array of observed series
            m - number of observations, has to be larger or equal than number 
                of lags
            k - number of series

        Returns
        -------
        X_forecast : numpy.ndarray
            (m-lags+1,k) array of forecasted series
            m - number of observations
            k - number of series
        """
        
        T, k = X.shape
        lags = self.lags
        if self.k != k:
            raise ValueError('The number of series differs between training and forecast data.')
            
        # construct regressor matrix
        Z1 = np.hstack(tuple([X[lag:T-lags+lag+1,:] for lag in range(lags)]))
        Z1 = sm.add_constant(Z1).T
        
        # forecast based on estimated model parameters
        X_pred = self.A1 @ Z1
        X_pred = X_pred.T
        
        return X_pred



class DFM():
    """
    Class to estimate a linear dynamic factor model via principal component
    analysis (PCA) and make predictions.
    
    Available methods:
    - train: train the linear factor model for given factor lags, idiosyncratic
        lags and factors .
    - train_CV: train the linear factor model via a grid search for the
        hyperparameters, choosing the model minimizing the validation MSE.
    - forecast: computes 1-step-ahead forecasts based on the estimated model.
    
    The method forecast can be used only after having estimated the model, i.e.
    after having run the method train or the method train_CV.
    """
    
    def __init__(self):
        self.factor_dynamics = VAR()
        self.noise_dynamics = diagonal_VAR()
    
    
    def train(self, X, 
              lags_factor_dynamics = 1, 
              lags_idiosyncratic_dynamics = 1, 
              number_of_factors = 2,
              method = 'standardized',
              verbose = True):
        """
        train(self, X, 
                  lags_factor_dynamics = 1, 
                  lags_idiosyncratic_dynamics = 1, 
                  number_of_factors = 2,
                  method = 'standardized',
                  verbose = True)
        
        Trains a linear dynamic factor model for the series X.

        Parameters
        ----------
        X : numpy.ndarray
            (T,k) array of observed series
            T - number of observations
            k - number of series
        lags_factor_dynamics : int, optional
            Number of lags to use for the estimation of the factor dynamics. 
            The default is 1.
        lags_idiosyncratic_dynamics : int, optional
            Number of lags to use for the estimation of the idiosyncratic noise
            dynamics. The default is 1.
        number_of_factors : int, optional
            Number of factors to estimate, i.e. number of neurons in the 
            bottleneck layer. The default is 2.
        method : str, optional
            Scaling before applying PCA. Should be one of:
            - 'original': no scaling, not recommended!
            - 'demeaned': demean the series before applying PCA
            - 'standardized': standardize the series (mean=0, std=1)
            The default is 'standardized'.
        verbose : bool, optional
            Indicates, whether updates on the training progress should be 
            outputted in the console. The default is True.

        Returns
        -------
        None.
        """
        
        self.verbose = verbose
        self.method = method
        
        T,k = X.shape
        self.lags_factor_dynamics = lags_factor_dynamics
        self.lags_idiosyncratic_dynamics = lags_idiosyncratic_dynamics
        
        # scale series
        self.X_mean = np.mean(X,0)
        self.X_std = np.std(X,0)
        if method == 'original':
            pass
        elif method == 'demeaned':
            X = X - self.X_mean
        elif method == 'standardized':
            X = (X - self.X_mean) / self.X_std
        else:
            raise ValueError("method should be 'original', 'demeaned' or 'standardized'")
        
        # Estimate factors via PCA
        XX = X.T @ X / T
        eig = np.linalg.eig(XX)
        order = np.argsort(eig[0])
        order = order[::-1]
        self.eigval = eig[0][order] # store eigenvalues as attribute
        eigvec = eig[1][:,order]
        self.L = eigvec[:,:number_of_factors].T
        F = X @ self.L.T
        
        # Estimate factor dynamics via VAR model
        self.factor_dynamics.train(F, lags = lags_factor_dynamics)
        
        # estimate idiosyncratic noise dynamics
        if self.lags_idiosyncratic_dynamics > 0:
            X_resid = X - F @ self.L
            self.noise_dynamics.train(X_resid, self.lags_idiosyncratic_dynamics)
    
    
    def train_CV(self, X, 
                 range_lags_fd = [i+1 for i in range(5)], 
                 range_lags_id = [i+1 for i in range(3)], 
                 range_numfac = [i+1 for i in range(5)],
                 method = 'standardized',
                 verbose = True):
        """
        train_CV(self, X, 
                      range_lags_fd = [i+1 for i in range(5)], 
                      range_lags_id = [i+1 for i in range(3)], 
                      range_numfac = [i+1 for i in range(5)],
                      method = 'standardized',
                      verbose = True)
        
        Trains a DFM for the series X. The last 10% of observations are used 
        for validation. Chooses the hyperparameters in the given ranges that 
        minimize the validation MSE. The best model is trained again on the 
        whole set.

        Parameters
        ----------
        X : numpy.ndarray
            (T,k) array of observed series
            T - number of observations
            k - number of series
        range_lags_fd : list, optional
            List of integers, number of lags to consider for the factor 
            dynamics. The default is [1,2,3,4,5].
        range_lags_id : list, optional
            List of integers, numbers of lags to consider for the idiosyncratic 
            dynamics. The default is [1,2,3].
        range_numfac : list, optional
            List of integers, numbers of factors to consider. The default is 
            [1,2,3,4,5].
        method : str, optional
            Scaling before applying PCA. Should be one of:
            - 'original': no scaling
            - 'demeaned': demean the series before applying PCA
            - 'standardized': standardize the series (mean=0, std=1)
            The default is 'standardized'.
        verbose : bool, optional
            Indicates, whether updates on the training progress should be 
            outputted in the console. The default is True.

        Returns
        -------
        None.
        """
        
        self.verbose = verbose
        
        T,k = X.shape
        split = int(np.round(0.9*T)) # index for splitting in training and validation
        X_train = X[:split, :] # training set
        X_val = X[split:, :] # validation set
        MSE = np.zeros((len(range_numfac), len(range_lags_id), len(range_lags_fd)))
        for i,num_fac in enumerate(range_numfac):
            for j,lags_id in enumerate(range_lags_id):
                for k,lags_fd in enumerate(range_lags_fd):
                    # print(str(i+1) + ' of ' + str(len(range_numfac)))
                    maxlags = max([lags_fd, lags_id])
                    self.train(X_train, lags_fd, lags_id, num_fac, method, verbose)
                    X_pred = self.forecast(X_val)
                    MSE[i,j,k] = np.mean((X_val[maxlags:, :] - X_pred[:-1, :])**2)
                    
        # obtain MSE optimal hyperparameters
        best_index = [x[0] for x in np.where(MSE == np.min(MSE))]
        self.best_numfac = range_numfac[best_index[0]]
        self.best_lags_id = range_lags_id[best_index[1]]
        self.best_lags_fd = range_lags_fd[best_index[2]]
        if verbose == True:
            print(MSE)
        
        # train the model for the number of lags minimizing the cross-val MSE
        self.train(X_train, 
                   self.best_lags_fd,
                   self.best_lags_id,
                   self.best_numfac)
    
    
    def forecast(self, X):
        """
        forecast(self, X)
        
        Computes 1-step-ahead forecasts for the provided observations based on
        the estimated model.

        Parameters
        ----------
        X : numpy.ndarray
            (m,k) array of observed series
            m - number of observations, has to be larger or equal than number 
                of lags
            k - number of series

        Returns
        -------
        X_pred : numpy.ndarray
            (m-lags+1,k) array of forecasted series
            m - number of observations
            k - number of series
        """
        
        # scale series
        if self.method == 'original':
            pass
        elif self.method == 'demeaned':
            X = X - self.X_mean
        elif self.method == 'standardized':
            X = (X - self.X_mean) / self.X_std
        else:
            raise ValueError("method should be 'original', 'demeaned' or 'standardized'")
        
        F = X @ self.L.T
        F_pred = self.factor_dynamics.forecast(F)
        X_predicted = F_pred @ self.L
        
        if self.lags_idiosyncratic_dynamics > 0:
            # obtain residuals to predict residual dynamics
            e = X - F @ self.L
            e_predicted = self.noise_dynamics.forecast(e)
            
            # merge factor and noise components
            lags_diff = self.lags_idiosyncratic_dynamics - self.lags_factor_dynamics
            if lags_diff >= 0:
                X_predicted = X_predicted[lags_diff:,:] + e_predicted
            else:
                X_predicted += e_predicted[-lags_diff:,:]
        
        # rescale to predicted series
        if self.method == 'original':
            X_pred = X_predicted
        elif self.method == 'demeaned':
            X_pred = X_predicted + self.X_mean
        elif self.method == 'standardized':
            X_pred = X_predicted * self.X_std + self.X_mean
        else:
            raise ValueError("method should be 'original', 'demeaned' or 'standardized'")
        
        return X_pred