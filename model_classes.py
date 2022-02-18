import numpy as np
import keras
from keras.backend import clear_session

import statsmodels.api as sm
from numpy.linalg import inv
import time



class autoencoder(keras.models.Sequential):
    def train(self, X_raw, number_of_factors):
        """
        X_raw: np.ndarray
            (T, k) array of observed series
        number_of_factors: int
            number of factors to estimate, neurons in the bottleneck layer
        """
        # obtain number of observations and series
        T, k = X_raw.shape
        
        # standardize input
        self.X_mean = np.mean(X_raw, 0)
        self.X_std = np.std(X_raw, 0)
        X = (X_raw - self.X_mean) / self.X_std
        
        # Model Architecture
        self.add(keras.layers.Dense(100, 
                                    activation = 'tanh',
                                    input_dim = k,
                                    kernel_regularizer = keras.regularizers.l2(0.001)))
        self.add(keras.layers.Dense(number_of_factors))
        self.add(keras.layers.Dense(100,
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
        # standardize series
        X = (X_raw - self.X_mean) / self.X_std
        
        # obtain encoder
        encoder = keras.models.Sequential()
        for layer in self.layers[:2]:
            encoder.add(layer)
        
        # return encoded series
        return encoder.predict(X)
    
    def decode(self, F):
        # obtain decoder
        decoder = keras.models.Sequential()
        for layer in self.layers[2:]:
            decoder.add(layer)
        
        # return rescaled predicted series
        return decoder.predict(F) * self.X_std + self.X_mean


class TimeSeriesMLP():
    def __init__(self):
        pass
    def train(self, X_raw, lags=1):
        """
        X_raw: np.ndarray
            (T, k) array of observed series
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
        self.MLP.add(keras.layers.Dense(100, 
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
        
    def train_CV(self, X, range_lags = [i+1 for i in range(4)]):
        """
        Trains a MLP for the series X. Chooses the number of lags in the given 
        range that minimizes the cross-validation MSE. The last 10% of the 
        observations are used for validation. The best model is trained again 
        on the whole set.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        range_lags: list of int, optional
            Number of lags to consider. 
            The default is [1,2,3,4,5].
        """
        T,k = X.shape
        split = int(np.round(0.9*T)) # index for splitting in training and validation
        X_train = X[:split, :] # training set
        X_val = X[split:, :] # validation set
        MSE = np.zeros(len(range_lags))
        for i,lags in enumerate(range_lags):
            # print(str(i+1) + ' of ' + str(len(range_numfac)))
            clear_session()
            self.train(X_train, lags)
            X_pred = self.forecast(X_val)
            MSE[i] = np.mean((X_val[lags:, :] - X_pred[:-1, :])**2)
                    
        # obtain MSE optimal hyperparameters
        self.best_lags = range_lags[np.argmin(MSE)]
        print(MSE)
        
        # train the model for the number of lags minimizing the cross-val MSE
        self.train(X_train, self.best_lags)
    
    def forecast(self, X_raw):
        """
        X_raw: np.ndarray
            (T, k) array of observed series
        """
        # obtain length and number of series
        T, k = X_raw.shape
        
        # Standardize series
        X = (X_raw - self.X_mean) / self.X_std
        
        # Prepare Data
        X_x = np.hstack(tuple(X[i:T-self.lags+i+1,:] for i in range(self.lags)))
        
        # predict output
        X_pred = self.MLP.predict(X_x)
        
        # return rescaled variables
        return X_pred * self.X_std + self.X_mean



# class TimeSeriesMLP(keras.models.Sequential):
#     def train(self, X_raw, lags=1):
#         """
#         X_raw: np.ndarray
#             (T, k) array of observed series
#         """
#         T, k = X_raw.shape
#         self.lags = lags
        
#         # Standardize series
#         self.X_mean = np.mean(X_raw, 0)
#         self.X_std = np.std(X_raw, 0)
#         X = (X_raw - self.X_mean) / self.X_std
        
#         # Model architecture
#         self.add(keras.layers.Dense(100, 
#                                     activation = 'tanh',
#                                     input_dim = k * lags,
#                                     kernel_regularizer = keras.regularizers.l2(0.001)))
#         self.add(keras.layers.Dense(k))
        
#         # Compilation
#         self.compile(optimizer = keras.optimizers.Adam(lr=0.015, 
#                                                        beta_1=0.9, 
#                                                        beta_2=0.999, 
#                                                        epsilon=1E-8,
#                                                        amsgrad=False),
#                      loss = 'mse')
        
#         # Prepare Data
#         X_x = np.hstack(tuple(X[i:T-lags+i,:] for i in range(lags)))
#         X_y = X[lags:,:]
        
#         # Training
#         self.fit(X_x, X_y,
#                  epochs = 300,
#                  batch_size = 512,
#                  validation_split = 0.1,
#                  callbacks = [keras.callbacks.EarlyStopping(patience = 50,
#                                                             restore_best_weights = True)],
#                  verbose = 0)
        
#     def train_CV(self, X, range_lags = [i+1 for i in range(4)]):
#         """
#         Trains a MLP for the series X. Chooses the number of lags in the given 
#         range that minimizes the cross-validation MSE. The last 10% of the 
#         observations are used for validation. The best model is trained again 
#         on the whole set.

#         Parameters
#         ----------
#         X : TYPE
#             DESCRIPTION.
#         range_lags: list of int, optional
#             Number of lags to consider. 
#             The default is [1,2,3,4,5].
#         """
#         T,k = X.shape
#         split = int(np.round(0.9*T)) # index for splitting in training and validation
#         X_train = X[:split, :] # training set
#         X_val = X[split:, :] # validation set
#         MSE = np.zeros(len(range_lags))
#         for i,lags in enumerate(range_lags):
#             # print(str(i+1) + ' of ' + str(len(range_numfac)))
#             self.train(X_train, lags)
#             X_pred = self.forecast(X_val)
#             MSE[i] = np.mean((X_val[lags:, :] - X_pred[:-1, :])**2)
                    
#         # obtain MSE optimal hyperparameters
#         self.best_lags = range_lags[np.argmin(MSE)]
#         print(MSE)
        
#         # train the model for the number of lags minimizing the cross-val MSE
#         self.train(X_train, self.best_lags)
    
#     def forecast(self, X_raw):
#         """
#         X_raw: np.ndarray
#             (T, k) array of observed series
#         """
#         # obtain length and number of series
#         T, k = X_raw.shape
        
#         # Standardize series
#         X = (X_raw - self.X_mean) / self.X_std
        
#         # Prepare Data
#         X_x = np.hstack(tuple(X[i:T-self.lags+i+1,:] for i in range(self.lags)))
        
#         # predict output
#         X_pred = self.predict(X_x)
        
#         # return rescaled variables
#         return X_pred * self.X_std + self.X_mean



class diagonal_VAR():
    def __init__(self):
        pass
    def train(self, X, lags):
        T,k = X.shape
        beta = np.zeros((k,lags))
        for i in range(k):
            Yi = X[lags:,i]
            Xi = np.zeros((T-lags, lags))
            for lag in range(lags):
                Xi[:,lags-lag-1] = X[lag:T-lags+lag,i]
            beta[i,:] = (inv(Xi.T @ Xi) @ Xi.T @ Yi).flatten()
        self.lags = lags
        self.k = k
        self.beta = beta
    
    def forecast(self, X):
        T,k = X.shape
        lags = self.lags
        if k != self.k:
            raise ValueError('The number of series differ between training and forecast data.')
        X_pred = np.zeros((T-lags+1, k))
        for i in range(k):
            Xi = np.zeros((T-lags+1, lags))
            for lag in range(lags):
                Xi[:,lags-lag-1] = X[lag:T-lags+lag+1,i]
            X_pred[:,i] = (Xi @ self.beta[i,:].reshape((-1,1))).flatten()
        
        return X_pred



class nDFM():
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
        Trains a nDFM for the series X.
        NOTE: The keras session is cleared to speed up the training. This might 
        result in problems if you have other keras models that you want to use 
        later on.

        Parameters
        ----------
        X: np.ndarray
            (T, k) array of observed series
        lags_factor_dynamics: int
            number of lags to use for the estimation of the factor dynamics
        lags_idiosyncratic_dynamics: int
            number of lags to use for the estimation of the idiosyncratic 
            noise dynamics
        number_of_factors: int
            number of factors to estimate, neurons in the bottleneck layer
        training_method: str
            method used for training, should be either 'state_space' or 'VAR'
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
        elif (training_method == 'VAR'):
            self.__train_VAR__(X)
        else:
            raise ValueError('The training method should be state_space of VAR')
    
    def train_CV(self, X, 
                  range_lags_fd = [i+1 for i in range(5)], 
                  range_lags_id = [i+1 for i in range(3)], 
                  range_numfac = [i+1 for i in range(5)],
                  training_method = 'state_space',
                  verbose = True):
        """
        Trains a nDFM for the series X. Chooses the hyperparameters in the 
        given ranges that minimize the cross-validation MSE. The last 10% of 
        the observations are used for validation. The best model is trained
        again on the whole set.
        NOTE: The keras session is cleared in each iteration to speed up the 
        training. This might result in problems if you have other keras models
        that you want to use later on.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        range_lags_fd : list of int, optional
            Number of lags to consider for the factor dynamics. 
            The default is [1,2,3,4,5].
        range_lags_id : list of int, optional
            Number of lags to consider for the idiosyncratic dynamics. 
            The default is [1,2,3].
        range_numfac : list of int, optional
            Number of factors to consider. The default is [1,2,3,4,5].
        training_method: str, optional
            method used for training, should be either 'state_space' or 'VAR'.
            The default is 'state_space'.
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
        X_raw: np.ndarray
            (T, k) array of observed series
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
    
    def __train_var__(self, 
                      X_raw, 
                      lags_factor_dynamics, 
                      lags_idiosyncratic_dynamics, 
                      number_of_factors):
        """
        X_raw: np.ndarray
            (T, k) array of observed series
        """
        pass
    
    def forecast(self, X):
        """
        X_raw: np.ndarray
            (T, k) array of observed series
        """
        try:
            if (self.training_method == 'state_space'):
                return self.__forecast_state_space__(X)
            elif (self.training_method == 'VAR'):
                return self.__forecast_VAR__(X)
            else:
                raise ValueError('Training method unknown.')
        except:
            raise ValueError('Train the model before forecasting')
    
    def __forecast_state_space__(self, X):
        """
        X_: np.ndarray
            (T, k) array of observed series
        """
        # estimate factors
        F = self.autoencoder.encode(X)
        
        # predict factors
        F_predicted = self.factor_dynamics.forecast(F)
        
        # predict series
        X_predicted = self.autoencoder.decode(F_predicted) #+X_resid_predicted
        
        # bootstrapped expectation
        # t1 = time.time()
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
        # print(time.time()-t1)
        
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



class VAR():
    def __init__(self):
        pass
    def train(self, X, lags=1):
        """
        Trains a VAR model for the series X with the given number of lags.

        Parameters
        ----------
        X : np.ndarray
            (T,k) array of series.
        lags : int, optional
            number of lags to consider. The default is 1.
        """
        T, k = X.shape
        self.k = k
        self.lags = lags
        
        Y1 = X[lags:,:].T
        Z1 = np.hstack(tuple([X[lag:T-lags+lag,:] for lag in range(lags)]))
        Z1 = sm.add_constant(Z1).T
        
        Y2 = Y1.reshape((-1,1))
        Z2 = np.kron(np.eye(k), Z1.T)
        # print(Z2.shape)
        A2 = inv(Z2.T @ Z2) @ Z2.T @ Y2
        A1 = A2.reshape((k, k*lags+1))
        self.A1 = A1
        
        A = [A1[:,0].reshape((-1,1))]
        for i in range(lags):
            A.append(A1[:,i*k+1:(i+1)*k+1])
        self.A = A
    
    def train_CV(self, X, range_lags = [i+1 for i in range(3)]):
        """
        Trains a VAR model for the series X. Chooses the lag in the given 
        range that minimizes the cross-validation MSE. The last 10% of the 
        observations are used for validation.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        range_lags : TYPE, optional
            DESCRIPTION. The default is [i for i in range(10)].
        """
        T,k = X.shape
        split = int(np.round(0.9*T)) # index for splitting in training and validation
        X_train = X[:split, :] # training set
        X_val = X[split:, :] # validation set
        MSE = np.zeros(len(range_lags))
        for i,lags in enumerate(range_lags):
            print(str(i+1) + ' of ' + str(len(range_lags)))
            self.train(X_train, lags = lags)
            X_pred = self.forecast(X_val)
            MSE[i] = np.mean((X_val[lags:, :] - X_pred[:-1, :])**2)
        self.best_lags = range_lags[np.argmin(MSE)]
        print(MSE)
        
        # train the model for the number of lags minimizing the cross-val MSE
        self.train(X, lags = self.best_lags)
    
    def forecast(self, X):
        T, k = X.shape
        lags = self.lags
        if self.k != k:
            raise ValueError('The number of series differ between training and forecast data.')
        Z1 = np.hstack(tuple([X[lag:T-lags+lag+1,:] for lag in range(lags)]))
        Z1 = sm.add_constant(Z1).T
        
        X_pred = self.A1 @ Z1
        X_pred = X_pred.T
        
        return X_pred


# neither centered, nor standardized
class DFM():
    """
    
    """
    def __init__(self):
        self.factor_dynamics = VAR()
        self.noise_dynamics = diagonal_VAR()
    
    def train(self, X, 
              lags_factor_dynamics = 1, 
              lags_idiosyncratic_dynamics = 1, 
              number_of_factors = 2,
              verbose = True):
        """
        

        Parameters
        ----------
        X : np.ndarray
            (T,k) array, series.
        lags_factor_dynamics : int, optional
            DESCRIPTION. The default is 1.
        lags_idiosyncratic_dynamics : int, optional
            DESCRIPTION. The default is 1.
        number_of_factors : int, optional
            DESCRIPTION. The default is 2.
        """
        self.verbose = verbose
        
        T,k = X.shape
        self.lags_factor_dynamics = lags_factor_dynamics
        self.lags_idiosyncratic_dynamics = lags_idiosyncratic_dynamics
        
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
                 verbose = True):
        """
        Trains a DFM for the series X. Chooses the number of factors in the 
        given range that minimizes the cross-validation MSE. The last 10% of 
        the observations are used for validation. The best model is trained
        again on the whole set.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        range_lags_fd : list of int, optional
            Number of lags to consider for the factor dynamics. 
            The default is [1,2,3,4,5].
        range_lags_id : list of int, optional
            Number of lags to consider for the idiosyncratic dynamics. 
            The default is [1,2,3].
        range_numfac : list of int, optional
            Number of factors to consider. The default is [1,2,3,4,5].
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
                    self.train(X_train, lags_fd, lags_id, num_fac)
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
        
        return X_predicted


# centered, not standardized
class DFM2():
    """
    
    """
    def __init__(self):
        self.factor_dynamics = VAR()
        self.noise_dynamics = diagonal_VAR()
    
    def train(self, X, 
              lags_factor_dynamics = 1, 
              lags_idiosyncratic_dynamics = 1, 
              number_of_factors = 2,
              verbose = True):
        """
        

        Parameters
        ----------
        X : np.ndarray
            (T,k) array, series.
        lags_factor_dynamics : int, optional
            DESCRIPTION. The default is 1.
        lags_idiosyncratic_dynamics : int, optional
            DESCRIPTION. The default is 1.
        number_of_factors : int, optional
            DESCRIPTION. The default is 2.
        """
        self.verbose = verbose
        
        T,k = X.shape
        self.lags_factor_dynamics = lags_factor_dynamics
        self.lags_idiosyncratic_dynamics = lags_idiosyncratic_dynamics
        
        # demean series
        self.X_mean = np.mean(X,0)
        X -= self.X_mean
        
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
                 verbose = True):
        """
        Trains a DFM for the series X. Chooses the number of factors in the 
        given range that minimizes the cross-validation MSE. The last 10% of 
        the observations are used for validation. The best model is trained
        again on the whole set.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        range_lags_fd : list of int, optional
            Number of lags to consider for the factor dynamics. 
            The default is [1,2,3,4,5].
        range_lags_id : list of int, optional
            Number of lags to consider for the idiosyncratic dynamics. 
            The default is [1,2,3].
        range_numfac : list of int, optional
            Number of factors to consider. The default is [1,2,3,4,5].
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
                    self.train(X_train, lags_fd, lags_id, num_fac)
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
        # demean series
        X -= self.X_mean
        
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
        
        # add mean to predicted series
        X_predicted += self.X_mean
        
        return X_predicted


# centered and standardized
class DFM3():
    """
    
    """
    def __init__(self):
        self.factor_dynamics = VAR()
        self.noise_dynamics = diagonal_VAR()
    
    def train(self, X, 
              lags_factor_dynamics = 1, 
              lags_idiosyncratic_dynamics = 1, 
              number_of_factors = 2,
              verbose = True):
        """
        

        Parameters
        ----------
        X : np.ndarray
            (T,k) array, series.
        lags_factor_dynamics : int, optional
            DESCRIPTION. The default is 1.
        lags_idiosyncratic_dynamics : int, optional
            DESCRIPTION. The default is 1.
        number_of_factors : int, optional
            DESCRIPTION. The default is 2.
        """
        self.verbose = verbose
        
        T,k = X.shape
        self.lags_factor_dynamics = lags_factor_dynamics
        self.lags_idiosyncratic_dynamics = lags_idiosyncratic_dynamics
        
        # demean series
        self.X_mean = np.mean(X,0)
        self.X_std = np.std(X,0)
        X = (X - self.X_mean) / self.X_std
        
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
                 verbose = True):
        """
        Trains a DFM for the series X. Chooses the number of factors in the 
        given range that minimizes the cross-validation MSE. The last 10% of 
        the observations are used for validation. The best model is trained
        again on the whole set.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        range_lags_fd : list of int, optional
            Number of lags to consider for the factor dynamics. 
            The default is [1,2,3,4,5].
        range_lags_id : list of int, optional
            Number of lags to consider for the idiosyncratic dynamics. 
            The default is [1,2,3].
        range_numfac : list of int, optional
            Number of factors to consider. The default is [1,2,3,4,5].
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
                    self.train(X_train, lags_fd, lags_id, num_fac)
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
        # demean series
        X = (X - self.X_mean) / self.X_std
        
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
        
        # add mean to predicted series
        X_predicted = X_predicted * self.X_std + self.X_mean
        
        return X_predicted