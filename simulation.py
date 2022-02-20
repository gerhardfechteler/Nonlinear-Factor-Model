import numpy as np
from nDFM_module import nDFM, TimeSeriesMLP, VAR, DFM, nDFM_simulator
from keras.backend import clear_session
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf

# np.random.seed(2)


###############################################################################
# Data generation

# series properties
r = 5
k = 50
T_train = 1000
T_test = 1000
lags_factor_dynamics = 2
lags_idiosyncratic_dynamics = 1
p_nonlin = 0.8
signal_noise_ratio = 2

maxlags = max([lags_factor_dynamics, lags_idiosyncratic_dynamics])

# Simulate series
simulator = nDFM_simulator(k, r, T_test + T_train, 
                           lags_factor_dynamics, lags_idiosyncratic_dynamics,
                           p_nonlin, signal_noise_ratio)
simulation = simulator.simulate()
X = simulation['X']

# Create train and test data
X_train = X[:T_train,:]
X_test = X[T_train:,:]


###############################################################################
# Visualization of properties of simulated series

# Factor and series correlation table
F_corr = pd.DataFrame(simulation['factors']).corr()
X_corr = pd.DataFrame(simulation['X']).corr()

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

# Plot autocorrelations for factors and series
plt.figure(figsize=(10,7))
lags=10
for i in range(5):
    plt.subplot(3,5,i+1)
    ax = plt.gca()
    plot_pacf(simulation['factors'][:,i], ax=ax, method='ywm', lags=lags)
    plt.ylim([-0.2,1.1])
    plt.title('$F_{{{},t}}$'.format(i+1))
    if i==0:
        plt.ylabel('Partial autocorrelation')
for i in range(5):
    plt.subplot(3,5,5+i+1)
    ax = plt.gca()
    plot_pacf(simulation['idiosyncratic_noise'][:,i], ax=ax, method='ywm', lags=lags)
    plt.ylim([-0.2,1.1])
    plt.title('$\epsilon_{{{},t}}$'.format(i+1))
    if i==0:
        plt.ylabel('Partial autocorrelation')
for i in range(5):
    plt.subplot(3,5,10+i+1)
    ax = plt.gca()
    plot_pacf(simulation['X'][:,i], ax=ax, method='ywm', lags=lags)
    plt.ylim([-0.2,1.1])
    plt.title('$X_{{{},t}}$'.format(i+1))
    plt.xlabel('Lag')
    if i==0:
        plt.ylabel('Partial autocorrelation')
plt.suptitle('Partial autocorrelation plots for factors, idiosyncratic noise \
             and series\n factor lags = {}, idisyncratic noise lags = {}, \
             signal-noise ratio = {}, $r={}$, $k={}$ (first 5 plotted), \
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

# Plot transformation functions
def trafo(x,ind,p=1):
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
trafo_text_list = [
    '$sgn(x)*ln(1+|x|)$',
    '$x-sgn(x)*ln(1+|x|)$',
    '$x+0.3*sgn(x)*|x|^2$',
    '$sgn(x)*min(1,|x|)$',
    '$sgn(x)*(1+|x|)$']
x_trafo = np.linspace(-3,3,1000)
plt.figure(figsize=(10,5))
plt.plot(x_trafo,x_trafo, '--k', label='linear')
for i in range(5):
    plt.plot(x_trafo, trafo(x_trafo, i), label=trafo_text_list[i])
plt.legend()
plt.ylabel('Transformation of x')
plt.xlabel('x')
plt.title('Transformations used in the function from factors $F$ to series $X$')
plt.tight_layout()


###############################################################################
# Estimation (on training data) and prediction (on test data)

# Oracle
X_pred_oracle, X_pred_oracle_boot = simulator.predict_oracle()
X_pred_oracle = X_pred_oracle[T_train:,:]
X_pred_oracle_boot = X_pred_oracle_boot[T_train:,:]
print('Oracle completed')

# # VAR
# model_VAR = VAR()
# model_VAR.train(X_train, maxlags)
# X_pred_VAR = model_VAR.forecast(X_test)
# print('VAR completed')

# # VAR cross-validated
# model_VAR_CV = VAR()
# model_VAR_CV.train_CV(X_train)
# X_pred_VAR_CV = model_VAR_CV.forecast(X_test)
# print('VAR_CV completed')

# # DFM
# model_DFM = DFM()
# model_DFM.train(X_train, lags_factor_dynamics, lags_idiosyncratic_dynamics, r)
# X_pred_DFM = model_DFM.forecast(X_test)
# print('DFM completed')

# # DFM cross-validated
# model_DFM_CV = DFM()
# model_DFM_CV.train_CV(X_train, 
#                       range_lags_id = [lags_idiosyncratic_dynamics],
#                       range_numfac = [i+1 for i in range(10)])
# X_pred_DFM_CV = model_DFM_CV.forecast(X_test)
# print('DFM_CV completed')

# # DFM2
# model_DFM2 = DFM2()
# model_DFM2.train(X_train, lags_factor_dynamics, lags_idiosyncratic_dynamics, r)
# X_pred_DFM2 = model_DFM2.forecast(X_test)
# print('DFM2 completed')

# # DFM2 cross-validated
# model_DFM2_CV = DFM2()
# model_DFM2_CV.train_CV(X_train, 
#                       range_lags_id = [lags_idiosyncratic_dynamics],
#                       range_numfac = [i+1 for i in range(10)])
# X_pred_DFM2_CV = model_DFM2_CV.forecast(X_test)
# print('DFM2_CV completed')

# DFM3
model_DFM3 = DFM()
model_DFM3.train(X_train, lags_factor_dynamics, lags_idiosyncratic_dynamics, r)
X_pred_DFM3 = model_DFM3.forecast(X_test)
print('DFM3 completed')

# DFM3 cross-validated
model_DFM3_CV = DFM()
model_DFM3_CV.train_CV(X_train, 
                       range_lags_fd=[lags_factor_dynamics],
                       range_lags_id = [lags_idiosyncratic_dynamics],
                       range_numfac = [i+1 for i in range(10)])
X_pred_DFM3_CV = model_DFM3_CV.forecast(X_test)
print('DFM3_CV completed')

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

# # MLP
# model_MLP = TimeSeriesMLP()
# model_MLP.train(X_train, maxlags)
# X_pred_MLP = model_MLP.forecast(X_test)
# print('MLP completed')

# # MLP cross-validated
# model_MLP_CV = TimeSeriesMLP()
# model_MLP_CV.train_CV(X_train)
# X_pred_MLP_CV = model_MLP_CV.forecast(X_test)
# print('MLP_CV completed')


###############################################################################
# Performance evaluation via validation MSE

# MSE comparison
MSE_oracle = np.mean((X_test[maxlags:]-X_pred_oracle[:-1])**2)
MSE_oracle_boot = np.mean((X_test[maxlags:]-X_pred_oracle_boot[:-1])**2)
MSE_nDFM = np.mean((X_test[maxlags:]-X_pred_nDFM[:-1])**2)
MSE_nDFM_boot = np.mean((X_test[maxlags:]-X_pred_nDFM_boot[:-1])**2)
# MSE_nDFM_CV = np.mean((X_test[-X_pred_nDFM_CV.shape[0]+1:]-X_pred_nDFM_CV[:-1])**2)
# MSE_nDFM_boot_CV = np.mean((X_test[-X_pred_nDFM_boot_CV.shape[0]+1:]-X_pred_nDFM_boot_CV[:-1])**2)
# MSE_DFM = np.mean((X_test[maxlags:]-X_pred_DFM[:-1])**2)
# MSE_DFM_CV = np.mean((X_test[-X_pred_DFM_CV.shape[0]+1:]-X_pred_DFM_CV[:-1])**2)
# MSE_DFM2 = np.mean((X_test[maxlags:]-X_pred_DFM2[:-1])**2)
# MSE_DFM2_CV = np.mean((X_test[-X_pred_DFM2_CV.shape[0]+1:]-X_pred_DFM2_CV[:-1])**2)
MSE_DFM3 = np.mean((X_test[maxlags:]-X_pred_DFM3[:-1])**2)
MSE_DFM3_CV = np.mean((X_test[-X_pred_DFM3_CV.shape[0]+1:]-X_pred_DFM3_CV[:-1])**2)
# MSE_MLP = np.mean((X_test[maxlags:]-X_pred_MLP[:-1])**2)
# MSE_MLP_CV = np.mean((X_test[-X_pred_MLP_CV.shape[0]+1:]-X_pred_MLP_CV[:-1])**2)
# MSE_VAR = np.mean((X_test[maxlags:]-X_pred_VAR[:-1])**2)
# MSE_VAR_CV = np.mean((X_test[-X_pred_VAR_CV.shape[0]+1:]-X_pred_VAR_CV[:-1])**2)
MSE_naive = np.mean((X_test[1:]-X_test[:-1])**2)
MSE_mean = np.mean((X_test - np.mean(X_train))**2)

V_X = np.var(X_test[maxlags:],0)

MSE_rel_oracle = np.mean(np.mean((X_test[maxlags:]-X_pred_oracle[:-1])**2, 0) / V_X)
MSE_rel_oracle_boot = np.mean(np.mean((X_test[maxlags:]-X_pred_oracle_boot[:-1])**2, 0) / V_X)
MSE_rel_nDFM = np.mean(np.mean((X_test[maxlags:]-X_pred_nDFM[:-1])**2, 0) / V_X)
MSE_rel_nDFM_boot = np.mean(np.mean((X_test[maxlags:]-X_pred_nDFM_boot[:-1])**2, 0) / V_X)
# MSE_rel_nDFM_CV = np.mean(np.mean((X_test[-X_pred_nDFM_CV.shape[0]+1:]-X_pred_nDFM_CV[:-1])**2, 0) / V_X)
# MSE_rel_nDFM_boot_CV = np.mean(np.mean((X_test[-X_pred_nDFM_boot_CV.shape[0]+1:]-X_pred_nDFM_boot_CV[:-1])**2, 0) / V_X)
# MSE_rel_DFM = np.mean(np.mean((X_test[maxlags:]-X_pred_DFM[:-1])**2, 0) / V_X)
# MSE_rel_DFM_CV = np.mean(np.mean((X_test[-X_pred_DFM_CV.shape[0]+1:]-X_pred_DFM_CV[:-1])**2, 0) / V_X)
# MSE_rel_DFM2 = np.mean(np.mean((X_test[maxlags:]-X_pred_DFM2[:-1])**2, 0) / V_X)
# MSE_rel_DFM2_CV = np.mean(np.mean((X_test[-X_pred_DFM2_CV.shape[0]+1:]-X_pred_DFM2_CV[:-1])**2, 0) / V_X)
MSE_rel_DFM3 = np.mean(np.mean((X_test[maxlags:]-X_pred_DFM3[:-1])**2, 0) / V_X)
MSE_rel_DFM3_CV = np.mean(np.mean((X_test[-X_pred_DFM3_CV.shape[0]+1:]-X_pred_DFM3_CV[:-1])**2, 0) / V_X)
# MSE_rel_MLP = np.mean(np.mean((X_test[maxlags:]-X_pred_MLP[:-1])**2, 0) / V_X)
# MSE_rel_MLP_CV = np.mean(np.mean((X_test[-X_pred_MLP_CV.shape[0]+1:]-X_pred_MLP_CV[:-1])**2, 0) / V_X)
# MSE_rel_VAR = np.mean(np.mean((X_test[maxlags:]-X_pred_VAR[:-1])**2, 0) / V_X)
# MSE_rel_VAR_CV = np.mean(np.mean((X_test[-X_pred_VAR_CV.shape[0]+1:]-X_pred_VAR_CV[:-1])**2, 0) / V_X)
MSE_rel_naive = np.mean(np.mean((X_test[1:]-X_test[:-1])**2, 0) / V_X)
MSE_rel_mean = np.mean(np.mean((X_test - np.mean(X_train))**2, 0) / V_X)

# clear keras session
clear_session()



