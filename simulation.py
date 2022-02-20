import numpy as np
from nDFM_module import nDFM, TimeSeriesMLP, VAR, DFM, nDFM_simulator
from keras.backend import clear_session
import matplotlib.pyplot as plt

# np.random.seed(2)
###############################################################################
# Data generation

# series properties
r = 5
k = 50
T_train = 1000
T_test = 1000
lags_factor_dynamics = 2
lags_idiosyncratic_dynamics = 2
p_nonlin = 0.8
p_sparse = 0.1
signal_noise_ratio = 2

maxlags = max([lags_factor_dynamics, lags_idiosyncratic_dynamics])

# Simulate series
simulator = nDFM_simulator(k, r, T_test + T_train, 
                           lags_factor_dynamics, lags_idiosyncratic_dynamics,
                           p_nonlin, p_sparse, signal_noise_ratio)
simulation = simulator.simulate()
X = simulation['X']

plt.figure(figsize=(10,7))
plt.plot(simulation['X_no_noise'][100:200,0:10])

# Create train and test data
X_train = X[:T_train,:]
X_test = X[T_train:,:]


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



