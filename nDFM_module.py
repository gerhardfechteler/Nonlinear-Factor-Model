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
              width_autoencoder = 100,
              width_factor_dynamics = 100,
              verbose = True):
        """
        train(self, X, 
                  lags_factor_dynamics = 1, 
                  lags_idiosyncratic_dynamics = 1, 
                  number_of_factors = 2,
                  training_method = 'state_space',
                  width_autoencoder = 100,
                  width_factor_dynamics = 100,
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
        width_autoencoder : int, optional
            Width of the hidden layer in the autoencoder. The default is 100.
        width_factor_dynamics : int, optional
            Width of the hidden layer of the MLP estimating the factor 
            dynamics. The default is 100.
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
        self.width_autoencoder = width_autoencoder
        self.width_factor_dynamics = width_factor_dynamics
        
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
                  width_autoencoder = 100,
                  width_factor_dynamics = 100,
                  verbose = True):
        """
        train_CV(self, X, 
                      range_lags_fd = [i+1 for i in range(5)], 
                      range_lags_id = [i+1 for i in range(3)], 
                      range_numfac = [i+1 for i in range(5)],
                      training_method = 'state_space',
                      width_autoencoder = 100,
                      width_factor_dynamics = 100,
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
        width_autoencoder : int, optional
            Width of the hidden layer in the autoencoder. The default is 100.
        width_factor_dynamics : int, optional
            Width of the hidden layer of the MLP estimating the factor 
            dynamics. The default is 100.
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
                    self.train(X_train, lags_fd, lags_id, num_fac, 
                               training_method, width_autoencoder, 
                               width_factor_dynamics, verbose)
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
        self.autoencoder.train(X, self.number_of_factors, w=self.width_autoencoder)
        
        # estimate factors
        F = self.autoencoder.encode(X)
        
        # estimate factor dynamics
        self.factor_dynamics.train(F, self.lags_factor_dynamics, w=self.width_factor_dynamics)
        
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



class nDFM_simulator():
    """
    Class to initialize a nonlinear dynamic factor model (nDFM), simulate data
    and make oracle forecasts.
    
    Available methods:
    - simulate: Simulate data from a nDFM with specified structure
    - predict_oracle: make oracle 1-step-ahead forecasts for the simulated data
    
    The method predict_oracle can be used only after having simulated data via
    the method simulate.
    """
    
    def __init__(self, k = 50, r = 2, T = 1000,
                 lags_factor_dynamics = 1,
                 lags_idiosyncratic_dynamics = 1,
                 p_nonlin = 0.5,
                 signal_noise_ratio = 1):
        """
        __init__(self, k = 50, r = 2, T = 1000,
                     lags_factor_dynamics = 1,
                     lags_idiosyncratic_dynamics = 1,
                     p_nonlin = 0.5,
                     signal_noise_ratio = 1)
        
        Initializes the simulator for a nonlinear dynamic factor model.

        Parameters
        ----------
        k : int, optional
            number of series. The default is 50.
        r : int, optional
            number of underlying factors. The default is 2.
        T : int, optional
            number of observations. The default is 1000.
        lags_factor_dynamics : int, optional
            number of lags in the factor dynamics. The default is 1.
        lags_idiosyncratic_dynamics : int, optional
            number of lags in the idiosyncratic noise dynamics. The default is 1.
        p_nonlin: float, optional
            measure for nonlinearity, should be in [0,1]. The default is 0.5.
        signal_noise_ratio : float, optional
            Ratio of standard deviations of factor innovations and 
            idiosyncratic innovations. The default is 1.

        Returns
        -------
        None.
        """
        
        self.k = k
        self.r = r
        self.T = T
        self.lags_factor_dynamics = lags_factor_dynamics
        self.lags_idiosyncratic_dynamics = lags_idiosyncratic_dynamics
        self.p_nonlin = p_nonlin
        self.signal_noise_ratio = signal_noise_ratio
        
        # indicator, whether the model is initialized
        self.initialized = False 
    
    
    def simulate(self):
        """
        simulate(self)
        
        Simulates a (T,k) array of series based on the model specified at 
        creation of the nDFM_simulator instance. The model parameters 
        (autoregressive matrices and function from factors to series) are 
        randomly drawn every time the method simulate is called.

        Returns
        -------
        simulation_wrapper : dict
            Keys:
            'X' : numpy.ndarray
                (T,k) array of simulated series
            'X_no_noise' : numpy.ndarray
                (T,k) array of simulated series without idiosyncratic noise
            'idiosyncratic_innovations' : numpy.ndarray
                (T,k) array of idiosyncratic innovations
            'idiosyncratic_noise' : numpy.ndarray
                (T,k) array of idiosyncratic noise
            'factor_innovations' : numpy.ndarray
                (T,r) array of factor innovations
            'factors' : numpy.ndarray
                (T,r) array of factors
        """
        
        # initialize the model, i.e. fix the model dynamics
        self.__initialize__()
        
        # obtain the simulated data
        simulation_wrapper = self.__simulate__()
        
        return simulation_wrapper
    
    
    def __initialize__(self):
        """
        __initialize__(self)
        
        Initializes the simulation by defining phi, psi and the mapping from
        factors to series in accordance with the model structure specified at
        creation of the nDFM_simulator instance.
        
        Creates the attributes
        ----------------------
        self.psi: list
            list of self.lags_factor_dynamics (r,r) arrays, specifies the
            lag polynomial matrix psi for the factor dynamics
        self.phi: list
            list of self.lags_idiosyncratic_dynamics (k,k) arrays, specifies 
            the lag polynomial matrix phi for the idiosyncratic dynamics
        self.decoder: tuple
            tuple containing the transformations for each factor, the
            transformations for the interactions, the factors that should
            be interacted, the loadings for the factors, and the loadings for
            the factor matrix.
        """
        
        # define the lag polynomial matrix for factor dynamics
        is_stationary = False
        while is_stationary == False:
            psi = [np.eye(self.r) * 0.7/self.lags_factor_dynamics + np.random.uniform(-0.25/self.lags_factor_dynamics,
                                                                                      0.25/self.lags_factor_dynamics,
                                                                                      (self.r, self.r))
                   for lag in range(self.lags_factor_dynamics)]
            Phi = np.zeros((self.r * self.lags_factor_dynamics, self.r * self.lags_factor_dynamics))
            for i in range(self.lags_factor_dynamics):
                Phi[:self.r, i*self.r:(i+1)*self.r] = psi[i]
                if i == 0:
                    continue
                Phi[i*self.r:(i+1)*self.r, (i-1)*self.r:i*self.r] = np.eye(self.r)
            if np.max(np.abs(np.linalg.eig(Phi)[0])) < 0.9: # check stationarity of the process
                is_stationary = True
        self.psi = psi
        
        # define the diagonal lag polynomial matrix for idiosyncratic dynamics
        phi = [np.eye(self.k) * 0.8/self.lags_idiosyncratic_dynamics
               for lag in range(self.lags_idiosyncratic_dynamics)]
        self.phi = phi
        
        # making a draw from the random components in the factor to series trafo
        # X_it = sum_j=1^r a_i,j T(F_jt; v_ij) + b_ij T(F_jt F_l_ij,t; w_ij) + eps_it
        A = np.random.uniform(-2, 2, (self.k, self.r)) # loading of individual factors
        B = np.random.uniform(-2, 2, (self.k, self.r)) # loading of factor interactions
        V = np.random.randint(0, 5, (self.k, self.r)) # trafos of individual factors
        W = np.random.randint(0, 5, (self.k, self.r)) # trafos of factor interactions
        L = np.random.randint(0, self.r-1, (self.T, self.r)) # interacting factor
        for i in range(self.r):
            L[:,i] = L[:,i] + np.heaviside(L[:,i]-i,1) # avoids that a factor interacts with itself
        self.decoder = (A,B,V,W,L)
        
        # the model is now initialized
        self.initialized = True
    
    
    def __simulate__(self):
        """
        __simulate__(self)
        
        Simulates a (T,k) array of series based on the random initialization
        made in __initialize__.
        
        Creates the attributes
        ----------------------
        self.F : numpy.ndarray
            (T,r) array of simulated Factors
        self.eps : numpy.ndarray
            (T,k) array of simulated idiosyncratic noise
        
        Returns
        -------
        simulation_wrapper : dict
            Keys:
            'X' : numpy.ndarray
                (T,k) array of simulated series
            'X_no_noise' : numpy.ndarray
                (T,k) array of simulated series without idiosyncratic noise
            'idiosyncratic_innovations' : numpy.ndarray
                (T,k) array of idiosyncratic innovations
            'idiosyncratic_noise' : numpy.ndarray
                (T,k) array of idiosyncratic noise
            'factor_innovations' : numpy.ndarray
                (T,r) array of factor innovations
            'factors' : numpy.ndarray
                (T,r) array of factors
        """
        
        # Check, whether the model has been initialized
        if self.initialized == False:
            raise ValueError("Initalize the model before calling '__simulate__' or use 'simulate' instead.")
        
        # length of burnin period
        burnin = max(self.lags_factor_dynamics, self.lags_idiosyncratic_dynamics) + 100
        
        # simulate factors
        Xi = np.random.normal(0, 1, (self.T + burnin, self.r)) # factor innovations
        F = np.copy(Xi)
        for t in np.arange(self.lags_factor_dynamics, self.T + burnin):
            for lag in range(self.lags_factor_dynamics):
                F[t, :] += F[t-lag-1, :] @ self.psi[lag].transpose()
        self.F = F[burnin:,:] # factors
        
        # simulate noise dynamics
        u = np.random.normal(0, 
                             1/self.signal_noise_ratio, 
                             (self.T + burnin, self.k)) # idiosyncratic innovations
        eps = np.copy(u)
        for t in np.arange(self.lags_idiosyncratic_dynamics, self.T + burnin):
            for lag in range(self.lags_idiosyncratic_dynamics):
                eps[t,:] += eps[t-lag-1, :] @ self.phi[lag].T
        self.eps = eps[burnin:,:] # idiosyncratic noise
        
        # obtain series without noise
        X_no_noise = self.__F2X__(F[burnin:,:])
        
        # obtain final series
        X = X_no_noise + eps[burnin:,:]
        
        # wrap up simulated series
        simulation_wrapper = {
            'X' : X,
            'X_no_noise' : X_no_noise,
            'idiosyncratic_innovations' : u[burnin:,:],
            'idiosyncratic_noise' : eps[burnin:,:],
            'factor_innovations' : Xi[burnin:,:],
            'factors' : self.F
            }
        
        return simulation_wrapper
    
    
    def __F2X__(self, F):
        """
        __F2X__(self, F)
        
        Function from factors to series based on the random generation in 
        the method __initialize__.
        
        Parameters
        ----------
        F : numpy.ndarray
            (T,r) array of factors
            
        Returns
        -------
        X_no_noise : numpy.ndarray
            (T,k) array of series without idiosyncratic noise term
        """
        
        # transformations
        def trafo(x,ind,p=1):
            """
            trafo(x,ind,p=1)
            
            Implementation of the transformation functions used on factors and
            factor interactions.

            Parameters
            ----------
            x : numpy.ndarray
                Data to transform.
            ind : int
                index for the transformation, should be one of 0,1,2,3,4.
            p : float, optional
                degree of nonlinearity, should be in [0,1]. The default is 1.

            Returns
            -------
            numpy.ndarray
                array of the transformed data.
            """
            
            if ind==0:
                return (1-p)*x + p*(np.sign(x)*np.log(1+np.abs(x)))
            elif ind==1:
                return (1-p)*x + p*(x-np.sign(x)*np.log(np.abs(x)+1))
            elif ind==2:
                return (1-p)*x + p*(x+0.3*np.sign(x)*np.abs(x)**(2))
            elif ind==3:
                return (1-p)*x + p*(np.heaviside(1-np.abs(x),0)*x + (1-np.heaviside(1-np.abs(x),0))*np.sign(x))
            elif ind==4:
                return (1-p)*x + p*(-1 + x + np.heaviside(x,0)*2)
            else:
                raise ValueError('Index for transformation type out of range')
        
        # get draws for the parameters of the function from factors to series
        A,B,V,W,L = self.decoder
        
        # construct the series from the factors
        X_no_noise = np.zeros((F.shape[0], self.k))
        for i in range(self.k):
            for j in range(self.r):
                X_no_noise[:,i] += (A[i,j] * trafo(F[:,j], V[i,j], self.p_nonlin) + 
                                    self.p_nonlin * B[i,j] * trafo(F[:,j]*F[:,L[i,j]], W[i,j], self.p_nonlin))
        
        return X_no_noise
        
    
    def predict_oracle(self):
        """
        predict_oracle(self)
        
        Predicts based on the (T,r) array of true factors and the (r,r) lag 
        polynomial matrix psi the one-step-ahead factor vector and then based 
        on the true mapping from factors to series the one-step-ahead series.
        Quasi-simulates the de-biased version of the prediction, as the 
        function from factors to series is non-linear.

        Returns
        -------
        X_predicted : numpy.ndarray
            (T-lags+1,k) array of predicted series, not bias-corrected. lags is
            the maximum number of lags in idiosyncratic and factor dynamics.
        X_predicted_boot : numpy.ndarray
            (T-lags+1,k) array of predicted series, bias-corrected.lags is the 
            maximum number of lags in idiosyncratic and factor dynamics.
        """
        
        # forecast factors
        F_pred = np.zeros((self.T+1-self.lags_factor_dynamics, self.r))
        for lag in range(self.lags_factor_dynamics):
            F_pred += self.F[self.lags_factor_dynamics-lag-1:self.T-lag, :] @ self.psi[lag].T
        
        # obtain series from forecasted factors
        X_predicted = self.__F2X__(F_pred) # plug-in estimate
        X_predicted_boot = np.zeros((F_pred.shape[0],self.k)) # bootstrapped version
        for i in range(50):
            X_predicted_boot += self.__F2X__(F_pred + np.random.normal(0,1,F_pred.shape))
        X_predicted_boot /= 50
        
        # forecast errors
        eps_pred = np.zeros((self.T+1-self.lags_idiosyncratic_dynamics, self.k))
        for lag in range(self.lags_idiosyncratic_dynamics):
            eps_pred += self.eps[self.lags_idiosyncratic_dynamics-lag-1:self.T-lag, :] @ self.phi[lag].T
        
        # merge factor and noise components
        lags_diff = self.lags_idiosyncratic_dynamics - self.lags_factor_dynamics
        if lags_diff >= 0:
            X_predicted = X_predicted[lags_diff:,:] + eps_pred
            X_predicted_boot = X_predicted_boot[lags_diff:,:] + eps_pred
        else:
            X_predicted += eps_pred[-lags_diff:,:]
            X_predicted_boot += eps_pred[-lags_diff:,:]
        
        return X_predicted, X_predicted_boot
