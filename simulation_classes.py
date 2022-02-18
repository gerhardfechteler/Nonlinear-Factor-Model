import numpy as np

class nDFM_simulator():
    """
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
    p_nonlin: float,
        measure for nonlinearity, should be in [0,1]
    p_sparse: float
        measure for sparsity, should be larger than zero.
    """
    def __init__(self, k = 50, r = 2, T = 1000,
                 lags_factor_dynamics = 1,
                 lags_idiosyncratic_dynamics = 1,
                 p_nonlin = 0.5,
                 p_sparse = 0.5,
                 signal_noise_ratio = 1):
        self.k = k
        self.r = r
        self.T = T
        self.lags_factor_dynamics = lags_factor_dynamics
        self.lags_idiosyncratic_dynamics = lags_idiosyncratic_dynamics
        self.p_sparse = p_sparse
        self.p_nonlin = p_nonlin
        self.signal_noise_ratio = signal_noise_ratio
        self.initialized = False # indicates, whether the model is initialized
    
    def simulate(self):
        """
        Simulates a (T,k) array of series based on the model specified at 
        creation of the nDFM_simulator instance.
        
        Creates the attributes
        ----------------------
        self.F: np.ndarray
            (T,r) array of simulated Factors
        self.eps: np.ndarray
            (T,k) array of simulated error terms
        self.psi: list
            list of self.lags_factor_dynamics (r,r) arrays, specifies the
            lag polynomial matrix psi for the factor dynamics
        self.phi: list
            list of self.lags_idiosyncratic_dynamics (T,T) arrays, specifies 
            the lag polynomial matrix phi for the idiosyncratic dynamics
        self.decoder: tuple
            tuple containing the list with the factors in each series and the 
            list of transformations of these factors.

        Returns
        -------
        X : np.ndarray
            (T,k) array of simulated series.
        """
        
        self.__initialize__()
        return self.__simulate__()
    
    def __initialize__(self):
        """
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
            tuple containing the list with the factors in each series and the 
            list of transformations of these factors.
        """
        # define the lag polynomial matrix for factor dynamics
        is_stationary = False
        while is_stationary == False:
            psi = [np.eye(self.r) * 0.8/self.lags_factor_dynamics + np.random.uniform(-0.3/self.lags_factor_dynamics,
                                                                                      0.3/self.lags_factor_dynamics,
                                                                                      (self.r, self.r))
                   for lag in range(self.lags_factor_dynamics)]
            Phi = np.zeros((self.r * self.lags_factor_dynamics, self.r * self.lags_factor_dynamics))
            for i in range(self.lags_factor_dynamics):
                Phi[:self.r, i*self.r:(i+1)*self.r] = psi[i]
                if i == 0:
                    continue
                Phi[i*self.r:(i+1)*self.r, (i-1)*self.r:i*self.r] = np.eye(self.r)
            if np.max(np.abs(np.linalg.eig(Phi)[0])) < 0.98: # check stationarity of the process
                is_stationary = True
        self.psi = psi
        
        # define the diagonal lag polynomial matrix for idiosyncratic dynamics
        phi = [np.eye(self.k) * 0.8/self.lags_idiosyncratic_dynamics
               for lag in range(self.lags_idiosyncratic_dynamics)]
        self.phi = phi
        
        # making a draw from the random components in the factor to series trafo
        # probabilities for transformations (see transformations in method __F2X__)
        p_trafo = [1-self.p_nonlin, self.p_nonlin/2, self.p_nonlin/2]
        # randomly choose the number of summands in each series, the number of
        # factors in each summand and the transformation of each factor.
        series = []
        factor_trafos = []
        for i in range(self.k): # self.k number of series are generated
            series.append([])
            factor_trafos.append([])
            for j in range(np.random.poisson(np.round(1/self.p_sparse))+1): # number of summands per series
                series[i].append([])
                factor_trafos[i].append([])
                for k in range(np.random.choice(2, p=[1-self.p_nonlin, self.p_nonlin])+1): # 1 or 2 factors in each summand
                    series[i][j].append(np.random.choice(self.r))
                    factor_trafos[i][j].append(np.random.choice(3, p=p_trafo))
        self.decoder = (series, factor_trafos)
        
        # the model is now initialized
        self.initialized = True
    
    def __simulate__(self):
        """
        Simulates a (T,k) array of series based on the random initialization
        made in __initialize__.
        
        Creates the attributes
        ----------------------
        self.F: np.ndarray
            (T,r) array of simulated Factors
        self.eps: np.ndarray
            (T,k) array of simulated error terms
        
        Returns
        -------
        X : np.ndarray
            (T,k) array of simulated series.
        """
        # Check, whether the model has been initialized
        if self.initialized == False:
            raise ValueError("Initalize the model before calling '__simulate__' or use 'simulate' instead.")
        
        # length of burnin period
        burnin = max(self.lags_factor_dynamics, self.lags_idiosyncratic_dynamics) + 100
        
        # simulate factors
        eta = np.random.normal(0, 1, (self.T + burnin, self.r)) # factor innovations
        F = np.copy(eta)
        for t in np.arange(self.lags_factor_dynamics, self.T + burnin):
            for lag in range(self.lags_factor_dynamics):
                F[t, :] += F[t-lag-1, :] @ self.psi[lag].transpose()
        self.F = F[burnin:,:]
        
        # simulate noise dynamics
        u = np.random.normal(0, 1/self.signal_noise_ratio, (self.T + burnin, self.k))
        eps = np.copy(u)
        for t in np.arange(self.lags_idiosyncratic_dynamics, self.T + burnin):
            for lag in range(self.lags_idiosyncratic_dynamics):
                eps[t,:] += eps[t-lag-1, :] @ self.phi[lag].T
        self.eps = eps
        
        # obtain final series
        X = self.__F2X__(F[burnin:,:]) + eps[burnin:,:]
        
        return X
    
    def __F2X__(self, F):
        """
        function from factors to series
        
        Parameters
        ----------
        F: np.ndarray
            (T,r) array of factors
            
        Returns
        -------
        X: np.ndarray
            (T,k) array of series
        """
        
        # damper transformation
        damper = lambda x: np.sign(x) * np.log(1 + np.abs(x))
        
        # transformation of the factors or accumulated factors
        def transform(x, trafo):
            if trafo==0:
                return x
            elif trafo==1:
                return damper(x)
            elif trafo==2:
                return x**2
            else:
                raise ValueError('Index for transformation type out of range')
        
        # obtain factors and trafos
        series, factor_trafos = (self.decoder)
        
        X = []
        for i,summands in enumerate(series):
            summands_i = []
            for j,factors in enumerate(summands):
                factors_i = []
                for k,factor in enumerate(factors):
                    factors_i.append(transform(F[:,factor],factor_trafos[i][j][k]))
                product = np.product(factors_i,0)
                summands_i.append(product)
            summ = np.sum(summands_i,0)
            X.append(summ)
        X = np.array(X).T
        
        return X
        
    def predict_oracle(self):
        """
        Predicts based on the (T,r) array of true factors and the (r,r) lag 
        polynomial matrix psi the one-step-ahead factor vector and then based 
        on the true mapping from factors to series the one-step-ahead series.

        Returns
        -------
        X_predicted : np.ndarray
            DESCRIPTION.

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



class nDFM_simulator2():
    """
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
    p_nonlin: float,
        measure for nonlinearity, should be in [0,1]
    p_sparse: float
        measure for sparsity, should be larger than zero.
    """
    def __init__(self, k = 50, r = 2, T = 1000,
                 lags_factor_dynamics = 1,
                 lags_idiosyncratic_dynamics = 1,
                 p_nonlin = 0.5,
                 p_sparse = 0.5,
                 signal_noise_ratio = 1):
        self.k = k
        self.r = r
        self.T = T
        self.lags_factor_dynamics = lags_factor_dynamics
        self.lags_idiosyncratic_dynamics = lags_idiosyncratic_dynamics
        self.p_sparse = p_sparse
        self.p_nonlin = p_nonlin
        self.signal_noise_ratio = signal_noise_ratio
        self.initialized = False # indicates, whether the model is initialized
    
    def simulate(self):
        """
        Simulates a (T,k) array of series based on the model specified at 
        creation of the nDFM_simulator instance.
        
        Creates the attributes
        ----------------------
        self.F: np.ndarray
            (T,r) array of simulated Factors
        self.eps: np.ndarray
            (T,k) array of simulated error terms
        self.psi: list
            list of self.lags_factor_dynamics (r,r) arrays, specifies the
            lag polynomial matrix psi for the factor dynamics
        self.phi: list
            list of self.lags_idiosyncratic_dynamics (T,T) arrays, specifies 
            the lag polynomial matrix phi for the idiosyncratic dynamics
        self.decoder: tuple
            tuple containing the list with the factors in each series and the 
            list of transformations of these factors.

        Returns
        -------
        X : np.ndarray
            (T,k) array of simulated series.
        """
        
        self.__initialize__()
        return self.__simulate__()
    
    def __initialize__(self):
        """
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
            tuple containing the list with the factors in each series and the 
            list of transformations of these factors.
        """
        # define the lag polynomial matrix for factor dynamics
        is_stationary = False
        while is_stationary == False:
            psi = [np.eye(self.r) * 0.8/self.lags_factor_dynamics + np.random.uniform(-0.3/self.lags_factor_dynamics,
                                                                                      0.3/self.lags_factor_dynamics,
                                                                                      (self.r, self.r))
                   for lag in range(self.lags_factor_dynamics)]
            Phi = np.zeros((self.r * self.lags_factor_dynamics, self.r * self.lags_factor_dynamics))
            for i in range(self.lags_factor_dynamics):
                Phi[:self.r, i*self.r:(i+1)*self.r] = psi[i]
                if i == 0:
                    continue
                Phi[i*self.r:(i+1)*self.r, (i-1)*self.r:i*self.r] = np.eye(self.r)
            if np.max(np.abs(np.linalg.eig(Phi)[0])) < 0.98: # check stationarity of the process
                is_stationary = True
        self.psi = psi
        
        # define the diagonal lag polynomial matrix for idiosyncratic dynamics
        phi = [np.eye(self.k) * 0.8/self.lags_idiosyncratic_dynamics
               for lag in range(self.lags_idiosyncratic_dynamics)]
        self.phi = phi
        
        # making a draw from the random components in the factor to series trafo
        # probabilities for transformations (see transformations in method __F2X__)
        p_trafo = [1-self.p_nonlin, self.p_nonlin/2, self.p_nonlin/2]
        # randomly choose the number of summands in each series, the number of
        # factors in each summand and the transformation of each factor.
        series = []
        factor_trafos = []
        for i in range(self.k): # self.k number of series are generated
            series.append([])
            factor_trafos.append([])
            for j in range(np.random.poisson(np.round(1/self.p_sparse))+1): # number of summands per series
                series[i].append([])
                factor_trafos[i].append([])
                for k in range(np.random.choice(2, p=[1-self.p_nonlin, self.p_nonlin])+1): # 1 or 2 factors in each summand
                    series[i][j].append(np.random.choice(self.r))
                    factor_trafos[i][j].append(np.random.choice(3, p=p_trafo))
        self.decoder = (series, factor_trafos)
        
        # the model is now initialized
        self.initialized = True
    
    def __simulate__(self):
        """
        Simulates a (T,k) array of series based on the random initialization
        made in __initialize__.
        
        Creates the attributes
        ----------------------
        self.F: np.ndarray
            (T,r) array of simulated Factors
        self.eps: np.ndarray
            (T,k) array of simulated error terms
        
        Returns
        -------
        X : np.ndarray
            (T,k) array of simulated series.
        """
        # Check, whether the model has been initialized
        if self.initialized == False:
            raise ValueError("Initalize the model before calling '__simulate__' or use 'simulate' instead.")
        
        # length of burnin period
        burnin = max(self.lags_factor_dynamics, self.lags_idiosyncratic_dynamics) + 100
        
        # simulate factors
        eta = np.random.normal(0, 1, (self.T + burnin, self.r)) # factor innovations
        F = np.copy(eta)
        for t in np.arange(self.lags_factor_dynamics, self.T + burnin):
            for lag in range(self.lags_factor_dynamics):
                F[t, :] += F[t-lag-1, :] @ self.psi[lag].transpose()
        self.F = F[burnin:,:]
        
        # simulate noise dynamics
        u = np.random.normal(0, 1/self.signal_noise_ratio, (self.T + burnin, self.k))
        eps = np.copy(u)
        for t in np.arange(self.lags_idiosyncratic_dynamics, self.T + burnin):
            for lag in range(self.lags_idiosyncratic_dynamics):
                eps[t,:] += eps[t-lag-1, :] @ self.phi[lag].T
        self.eps = eps
        
        # obtain final series
        X = self.__F2X__(F[burnin:,:]) + eps[burnin:,:]
        
        return X
    
    def __F2X__(self, F):
        """
        function from factors to series
        
        Parameters
        ----------
        F: np.ndarray
            (T,r) array of factors
            
        Returns
        -------
        X: np.ndarray
            (T,k) array of series
        """
        
        # damper transformation
        damper = lambda x: np.sign(x) * np.log(1 + np.abs(x))
        
        # transformation of the factors or accumulated factors
        def transform(x, trafo):
            if trafo==0:
                return x
            elif trafo==1:
                return damper(x)
            elif trafo==2:
                return np.sign(x) * x**2
            else:
                raise ValueError('Index for transformation type out of range')
        
        # obtain factors and trafos
        series, factor_trafos = (self.decoder)
        
        X = []
        for i,summands in enumerate(series):
            summands_i = []
            for j,factors in enumerate(summands):
                factors_i = []
                for k,factor in enumerate(factors):
                    factors_i.append(transform(F[:,factor],factor_trafos[i][j][k]))
                product = np.product(factors_i,0)
                summands_i.append(product)
            summ = np.sum(summands_i,0)
            X.append(summ)
        X = np.array(X).T
        
        return X
        
    def predict_oracle(self):
        """
        Predicts based on the (T,r) array of true factors and the (r,r) lag 
        polynomial matrix psi the one-step-ahead factor vector and then based 
        on the true mapping from factors to series the one-step-ahead series.

        Returns
        -------
        X_predicted : np.ndarray
            DESCRIPTION.

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


class nDFM_simulator3():
    """
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
    p_nonlin: float,
        measure for nonlinearity, should be in [0,1]
    p_sparse: float
        measure for sparsity, should be larger than zero.
    """
    def __init__(self, k = 50, r = 2, T = 1000,
                 lags_factor_dynamics = 1,
                 lags_idiosyncratic_dynamics = 1,
                 p_nonlin = 0.5,
                 p_sparse = 0.5,
                 signal_noise_ratio = 1):
        self.k = k
        self.r = r
        self.T = T
        self.lags_factor_dynamics = lags_factor_dynamics
        self.lags_idiosyncratic_dynamics = lags_idiosyncratic_dynamics
        self.p_sparse = p_sparse
        self.p_nonlin = p_nonlin
        self.signal_noise_ratio = signal_noise_ratio
        self.initialized = False # indicates, whether the model is initialized
    
    def simulate(self):
        """
        Simulates a (T,k) array of series based on the model specified at 
        creation of the nDFM_simulator instance.

        Returns
        -------
        X : np.ndarray
            (T,k) array of simulated series.
        """
        
        self.__initialize__()
        return self.__simulate__()
    
    def __initialize__(self):
        """
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
            transformations for the interactions and the factors that should
            be interacted.
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
        trafo1 = np.random.randint(0, 5, (self.k, self.r)) # trafos of individual factors
        trafo2 = np.random.randint(0, 5, (self.k, self.r)) # trafos of factor interactions
        F_interact = np.random.randint(0, self.r-1, (self.T, self.r)) # factors for interaction
        for i in range(self.r):
            F_interact[:,i] = F_interact[:,i] + np.heaviside(F_interact[:,i]-i,1) # avoids that a factor interacts with itself
        self.decoder = (trafo1, trafo2, F_interact)
        
        # the model is now initialized
        self.initialized = True
    
    def __simulate__(self):
        """
        Simulates a (T,k) array of series based on the random initialization
        made in __initialize__.
        
        Creates the attributes
        ----------------------
        self.F: np.ndarray
            (T,r) array of simulated Factors
        self.eps: np.ndarray
            (T,k) array of simulated error terms
        
        Returns
        -------
        X : np.ndarray
            (T,k) array of simulated series.
        """
        # Check, whether the model has been initialized
        if self.initialized == False:
            raise ValueError("Initalize the model before calling '__simulate__' or use 'simulate' instead.")
        
        # length of burnin period
        burnin = max(self.lags_factor_dynamics, self.lags_idiosyncratic_dynamics) + 100
        
        # simulate factors
        eta = np.random.normal(0, 1, (self.T + burnin, self.r)) # factor innovations
        F = np.copy(eta)
        for t in np.arange(self.lags_factor_dynamics, self.T + burnin):
            for lag in range(self.lags_factor_dynamics):
                F[t, :] += F[t-lag-1, :] @ self.psi[lag].transpose()
        self.F = F[burnin:,:]
        
        # simulate noise dynamics
        u = np.random.normal(0, 1/self.signal_noise_ratio, (self.T + burnin, self.k))
        eps = np.copy(u)
        for t in np.arange(self.lags_idiosyncratic_dynamics, self.T + burnin):
            for lag in range(self.lags_idiosyncratic_dynamics):
                eps[t,:] += eps[t-lag-1, :] @ self.phi[lag].T
        self.eps = eps
        
        # obtain final series
        X = self.__F2X__(F[burnin:,:]) + eps[burnin:,:]
        
        return X
    
    def __F2X__(self, F):
        """
        function from factors to series
        
        Parameters
        ----------
        F: np.ndarray
            (T,r) array of factors
            
        Returns
        -------
        X: np.ndarray
            (T,k) array of series
        """
        # transformations
        def trafo(x,ind,p):
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
        
        # obtain factors and trafos
        trafo1, trafo2, F_interact = self.decoder
        X = np.zeros((F.shape[0], self.k))
        for i in range(self.k):
            for j in range(self.r):
                X[:,i] += (1-self.p_nonlin/2) * trafo(F[:,j], trafo1[i,j], self.p_nonlin)
                X[:,i] += self.p_nonlin/2 * trafo(F[:,j]*F[:,F_interact[i,j]], trafo2[i,j], self.p_nonlin)
        return X
        
    def predict_oracle(self):
        """
        Predicts based on the (T,r) array of true factors and the (r,r) lag 
        polynomial matrix psi the one-step-ahead factor vector and then based 
        on the true mapping from factors to series the one-step-ahead series.

        Returns
        -------
        X_predicted : np.ndarray
            DESCRIPTION.

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
    


class nDFM_simulator4():
    """
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
    p_nonlin: float,
        measure for nonlinearity, should be in [0,1]
    p_sparse: float
        measure for sparsity, should be larger than zero.
    """
    def __init__(self, k = 50, r = 2, T = 1000,
                 lags_factor_dynamics = 1,
                 lags_idiosyncratic_dynamics = 1,
                 p_nonlin = 0.5,
                 p_sparse = 0.5,
                 signal_noise_ratio = 1):
        self.k = k
        self.r = r
        self.T = T
        self.lags_factor_dynamics = lags_factor_dynamics
        self.lags_idiosyncratic_dynamics = lags_idiosyncratic_dynamics
        self.p_sparse = p_sparse
        self.p_nonlin = p_nonlin
        self.signal_noise_ratio = signal_noise_ratio
        self.initialized = False # indicates, whether the model is initialized
    
    def simulate(self):
        """
        Simulates a (T,k) array of series based on the model specified at 
        creation of the nDFM_simulator instance.

        Returns
        -------
        X : np.ndarray
            (T,k) array of simulated series.
        """
        
        self.__initialize__()
        return self.__simulate__()
    
    def __initialize__(self):
        """
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
            transformations for the interactions and the factors that should
            be interacted.
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
        trafo1 = np.random.randint(0, 5, (self.k, self.r)) # trafos of individual factors
        trafo2 = np.random.randint(0, 5, (self.k, self.r)) # trafos of factor interactions
        F_interact = np.random.randint(0, self.r-1, (self.T, self.r)) # factors for interaction
        for i in range(self.r):
            F_interact[:,i] = F_interact[:,i] + np.heaviside(F_interact[:,i]-i,1) # avoids that a factor interacts with itself
        self.decoder = (trafo1, trafo2, F_interact)
        
        # the model is now initialized
        self.initialized = True
    
    def __simulate__(self):
        """
        Simulates a (T,k) array of series based on the random initialization
        made in __initialize__.
        
        Creates the attributes
        ----------------------
        self.F: np.ndarray
            (T,r) array of simulated Factors
        self.eps: np.ndarray
            (T,k) array of simulated error terms
        
        Returns
        -------
        X : np.ndarray
            (T,k) array of simulated series.
        """
        # Check, whether the model has been initialized
        if self.initialized == False:
            raise ValueError("Initalize the model before calling '__simulate__' or use 'simulate' instead.")
        
        # length of burnin period
        burnin = max(self.lags_factor_dynamics, self.lags_idiosyncratic_dynamics) + 100
        
        # simulate factors
        eta = np.random.normal(0, 1, (self.T + burnin, self.r)) # factor innovations
        F = np.copy(eta)
        for t in np.arange(self.lags_factor_dynamics, self.T + burnin):
            for lag in range(self.lags_factor_dynamics):
                F[t, :] += F[t-lag-1, :] @ self.psi[lag].transpose()
        self.F = F[burnin:,:]
        
        # simulate noise dynamics
        u = np.random.normal(0, 1/self.signal_noise_ratio, (self.T + burnin, self.k))
        eps = np.copy(u)
        for t in np.arange(self.lags_idiosyncratic_dynamics, self.T + burnin):
            for lag in range(self.lags_idiosyncratic_dynamics):
                eps[t,:] += eps[t-lag-1, :] @ self.phi[lag].T
        self.eps = eps
        
        # obtain final series
        X = self.__F2X__(F[burnin:,:]) + eps[burnin:,:]
        
        return X
    
    def __F2X__(self, F):
        """
        function from factors to series
        
        Parameters
        ----------
        F: np.ndarray
            (T,r) array of factors
            
        Returns
        -------
        X: np.ndarray
            (T,k) array of series
        """
        # transformations
        def trafo(x,ind,p):
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
        
        # obtain factors and trafos
        trafo1, trafo2, F_interact = self.decoder
        X = np.zeros((F.shape[0], self.k))
        for i in range(self.k):
            for j in range(self.r):
                X[:,i] += (1-self.p_nonlin/2) * trafo(F[:,j], trafo1[i,j], self.p_nonlin)
                X[:,i] += self.p_nonlin/2 * trafo(F[:,j]*F[:,F_interact[i,j]], trafo2[i,j], self.p_nonlin)
        return X
        
    def predict_oracle(self):
        """
        Predicts based on the (T,r) array of true factors and the (r,r) lag 
        polynomial matrix psi the one-step-ahead factor vector and then based 
        on the true mapping from factors to series the one-step-ahead series.

        Returns
        -------
        X_predicted : np.ndarray
            DESCRIPTION.

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