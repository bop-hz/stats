import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
np.random.seed(0)
p = np.arange(1,11)
MSE_1nn = np.zeros(10)
MSE_LS = np.zeros(10)
N_trials =1000
N_datasize = 500
for ii in p:
    y_hat_1nn = np.zeros(N_trials)
    y_hat_ls = np.zeros(N_trials)
    #target = np.random.default_rng().normal(0,1,N_trials)
    target = np.random.default_rng().normal(0,1)
    for jj in np.arange(0,N_trials):
        y_train_off = np.random.default_rng().normal(0,1,(N_datasize))
        x_train = np.random.uniform(-1,1,(N_datasize,ii))
        y_train = x_train[:,0]+y_train_off
        norm_x = np.diag(x_train@x_train.T)
        qq = np.argmin(norm_x)
        y_hat_1nn[jj] = y_train[qq]
        x_train_l = np.hstack((np.ones((N_datasize,1)),x_train))
        bb = linalg.inv(x_train_l.T @ x_train_l )@x_train_l.T@(y_train.T)
        y_hat_ls[jj] = np.array([1]+[0]*ii).T@bb
    MSE_1nn[ii-1] = (y_hat_1nn-target)@(y_hat_1nn-target).T/N_trials
    MSE_LS[ii-1]= (y_hat_ls-target)@(y_hat_ls-target).T/N_trials
        
plt.plot(p,MSE_1nn) 
plt.plot(p,MSE_LS)