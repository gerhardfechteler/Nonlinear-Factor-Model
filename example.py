import numpy as np
from nDFM_module import nDFM, DFM, nDFM_simulator
from keras.backend import clear_session
import matplotlib.pyplot as plt


###############################################################################
# Data generation

# number of factors
r = 5 

# number of series
k = 50

# number of observations
T_train = 1000 # training set
T_test = 1000 # test set

# number of lags in factor and idiosyncratic noise dynamics
lags_factor_dynamics = 2
lags_idiosyncratic_dynamics = 1

# degree of nonlinearity
p_nonlin = 0.8

# signal to noise ratio, ratio of standard deviations of factor and 
# idiosyncratic noise innovations
signal_noise_ratio = 2

# maxlags = max([lags_factor_dynamics, lags_idiosyncratic_dynamics])

# Simulate data from the nonlinear dynamic factor model
simulator = nDFM_simulator(k, r, T_test + T_train, 
                           lags_factor_dynamics, 
                           lags_idiosyncratic_dynamics,
                           p_nonlin, 
                           signal_noise_ratio)
simulation = simulator.simulate()

# obtain the simulated series
X = simulation['X']

# Create train and test data
X_train = X[:T_train,:]
X_test = X[T_train:,:]


###############################################################################
# Visualization simulated series

# Plotting parameters update to enable LaTeX
plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

# Plot simulated data series
plt.figure(figsize=(10,13))
plt.subplot(3,1,1)
plt.plot(simulation['factors'][100:200,0:10])
plt.ylabel('Factors $F_t$')
plt.subplot(3,1,2)
plt.plot(simulation['idiosyncratic_noise'][100:200,0:10])
plt.ylabel('Idiosyncratic noise $\epsilon_t$')
plt.subplot(3,1,3)
plt.plot(simulation['X'][100:200,0:10])
plt.ylabel('Simulated series $X_t$')
plt.xlabel('Time t')
plt.suptitle('Simulated factors, idiosyncratic noise and series\n \
             factor lags = {}, idisyncratic noise lags = {}, \
             signal-noise ratio = {}, $r={}$, $k={}$ (first 10 plotted), \
             $p_{{nonlin}}={}$'
             .format(lags_factor_dynamics, lags_idiosyncratic_dynamics, 
                     signal_noise_ratio, r, k, p_nonlin))
plt.tight_layout()

# Scree plot / eigenvalues
X_std = (X - np.mean(X,0)) / np.std(X,0)
XX = X_std.T @ X_std / X.shape[0]
eig = np.linalg.eig(XX)
order = np.argsort(eig[0])[::-1]
eigval = eig[0][order]
plt.figure(figsize=(10,5))
plt.scatter(np.arange(1,k+1),eigval, label = "Eigenvalues of $X'X$ in descending order")
ylim = plt.gca().get_ylim()
plt.vlines(r, ylim[0], ylim[1], linestyles='--', color='k', label='True number of factors')
plt.ylim(ylim)
plt.xlabel('Number of factors')
plt.ylabel('Eigenvalues')
plt.title('Scree plot for the simulated series $X$\n \
          factor lags = {}, idisyncratic noise lags = {}, \
          signal-noise ratio = {}, $r={}$, $k={}$, \
          $p_{{nonlin}}={}$'
          .format(lags_factor_dynamics, lags_idiosyncratic_dynamics, 
                  signal_noise_ratio, r, k, p_nonlin))
plt.legend()
plt.tight_layout()


###############################################################################
# Estimation (on training data) and prediction (on test data)

# Oracle
X_pred_oracle, X_pred_oracle_boot = simulator.predict_oracle()
X_pred_oracle = X_pred_oracle[T_train:,:]
X_pred_oracle_boot = X_pred_oracle_boot[T_train:,:]
print('Oracle completed')

# DFM
model_DFM = DFM()
model_DFM.train(X_train, lags_factor_dynamics, lags_idiosyncratic_dynamics, r)
X_pred_DFM3 = model_DFM.forecast(X_test)
print('DFM completed')

# DFM cross-validated
model_DFM_CV = DFM()
model_DFM_CV.train_CV(X_train, 
                       range_lags_fd = [lags_factor_dynamics],
                       range_lags_id = [lags_idiosyncratic_dynamics],
                       range_numfac = [i+1 for i in range(10)])
X_pred_DFM3_CV = model_DFM_CV.forecast(X_test)
print('DFM_CV completed')

# nonlinear DFM
model_nDFM = nDFM()
model_nDFM.train(X_train, lags_factor_dynamics, lags_idiosyncratic_dynamics, r)
X_pred_nDFM, X_pred_nDFM_boot = model_nDFM.forecast(X_test)
print('nDFM completed')

# # nonlinear DFM cross-validated
# model_nDFM_CV = nDFM()
# model_nDFM_CV.train_CV(X_train, 
#                        range_lags_id = [lags_idiosyncratic_dynamics],
#                        range_lags_fd = [lags_factor_dynamics])
# X_pred_nDFM_CV, X_pred_nDFM_boot_CV = model_nDFM_CV.forecast(X_test)
# print('nDFM_CV completed')


###############################################################################
# Performance evaluation via validation MSE

maxlags = max([lags_factor_dynamics, lags_idiosyncratic_dynamics])

# MSE comparison
MSE_oracle = np.mean((X_test[maxlags:]-X_pred_oracle[:-1])**2)
MSE_oracle_boot = np.mean((X_test[maxlags:]-X_pred_oracle_boot[:-1])**2)
MSE_nDFM = np.mean((X_test[maxlags:]-X_pred_nDFM[:-1])**2)
MSE_nDFM_boot = np.mean((X_test[maxlags:]-X_pred_nDFM_boot[:-1])**2)
# MSE_nDFM_CV = np.mean((X_test[-X_pred_nDFM_CV.shape[0]+1:]-X_pred_nDFM_CV[:-1])**2)
# MSE_nDFM_boot_CV = np.mean((X_test[-X_pred_nDFM_boot_CV.shape[0]+1:]-X_pred_nDFM_boot_CV[:-1])**2)
MSE_DFM3 = np.mean((X_test[maxlags:]-X_pred_DFM3[:-1])**2)
MSE_DFM3_CV = np.mean((X_test[-X_pred_DFM3_CV.shape[0]+1:]-X_pred_DFM3_CV[:-1])**2)
MSE_naive = np.mean((X_test[1:]-X_test[:-1])**2)
MSE_mean = np.mean((X_test - np.mean(X_train))**2)

# obtain the variances of the series
V_X = np.var(X_test[maxlags:],0)

# compare the relative MSE, i.e. the average MSE per series, relative to the
# variance of the respective series
MSE_rel_oracle = np.mean(np.mean((X_test[maxlags:]-X_pred_oracle[:-1])**2, 0) / V_X)
MSE_rel_oracle_boot = np.mean(np.mean((X_test[maxlags:]-X_pred_oracle_boot[:-1])**2, 0) / V_X)
MSE_rel_nDFM = np.mean(np.mean((X_test[maxlags:]-X_pred_nDFM[:-1])**2, 0) / V_X)
MSE_rel_nDFM_boot = np.mean(np.mean((X_test[maxlags:]-X_pred_nDFM_boot[:-1])**2, 0) / V_X)
# MSE_rel_nDFM_CV = np.mean(np.mean((X_test[-X_pred_nDFM_CV.shape[0]+1:]-X_pred_nDFM_CV[:-1])**2, 0) / V_X)
# MSE_rel_nDFM_boot_CV = np.mean(np.mean((X_test[-X_pred_nDFM_boot_CV.shape[0]+1:]-X_pred_nDFM_boot_CV[:-1])**2, 0) / V_X)
MSE_rel_DFM3 = np.mean(np.mean((X_test[maxlags:]-X_pred_DFM3[:-1])**2, 0) / V_X)
MSE_rel_DFM3_CV = np.mean(np.mean((X_test[-X_pred_DFM3_CV.shape[0]+1:]-X_pred_DFM3_CV[:-1])**2, 0) / V_X)
MSE_rel_naive = np.mean(np.mean((X_test[1:]-X_test[:-1])**2, 0) / V_X)
MSE_rel_mean = np.mean(np.mean((X_test - np.mean(X_train))**2, 0) / V_X)

# clear keras session
clear_session()