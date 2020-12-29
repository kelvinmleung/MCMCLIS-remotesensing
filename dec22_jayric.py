
# generate linear model for jayanth and ricardo
import numpy as np
import matplotlib.pyplot as plt
from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis

import sys, os, json
sys.path.insert(0, '../isofit/')
import isofit
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.inversion.inverse import Inversion
from isofit.configs.configs import Config  



## SETUP ##
atm = [0.5,2.5]
#atm = [0.05,1.75]
setup = Setup(atm)

Xtrain = np.load('results/samples/MOG/X_train.npy')
Ytrain = np.load('results/samples/MOG/Y_train.npy')


wv, aqua = np.loadtxt('JayanthRicardo/aquatic_spectrum_resampled.txt').T
wv, veg = np.loadtxt('JayanthRicardo/vegetation_spectrum_resampled.txt').T
phi = np.load('results/Regression/MOG/phi.npy')

meanX = np.mean(Xtrain,0)
varX = np.var(Xtrain,0)
meanY = np.mean(Ytrain,0)
varY = np.var(Ytrain,0)

scaleX = (Xtrain - meanX) / np.sqrt(varX)
scaleY = (Ytrain - meanY) / np.sqrt(varY)

aqua = np.concatenate((aqua, np.array(atm)))
veg = np.concatenate((veg, np.array(atm)))


ABC = 'B'


if ABC == 'A':
    truth = 0.3 * veg + 0.7 * aqua
elif ABC == 'B':
    truth = 0.7 * veg + 0.3 * aqua
elif ABC == 'C':
    truth = 0.5 * veg + 0.5 * aqua

r = Regression(setup)
a = Analysis(setup, r)


radiance = setup.fm.calc_rdn(truth, setup.geom)
noisecov = setup.fm.Seps(truth, radiance, setup.geom)
eps = np.random.multivariate_normal(np.zeros(len(radiance)), noisecov)
radNoisy = radiance + eps

# Isofit Inversion
inversion_settings = {"implementation": {
        "mode": "inversion",
        "inversion": {
        "windows": [[380.0, 1300.0], [1450, 1780.0], [1950.0, 2450.0]]}}}
inverse_config = Config(inversion_settings)
iv = Inversion(inverse_config, setup.fm)
state_trajectory = iv.invert(radNoisy, setup.geom)
state_est = state_trajectory[-1]
rfl_est, rdn_est, path_est, S_hat, K, G = iv.forward_uncertainty(state_est, radNoisy, setup.geom)


# prior index 5 (from surface_multicomp)
mu_x, gamma_x = setup.getPrior(idx_pr=5)


error = scaleY - scaleX @ phi.T
gamma_ygx_tilde = np.cov(error.T)
sigma_x_power = np.diag(varX ** -0.5)
sigma_y_power = np.diag(varY ** 0.5)
gamma_ygx = np.real(sigma_y_power @ gamma_ygx_tilde @ sigma_y_power)
phiFull = np.real(sigma_y_power @ phi @ sigma_x_power)

gamma_y = phiFull @ gamma_x @ phiFull.T + gamma_ygx
gamma_xgy = (np.identity(a.nx) - gamma_x @ phiFull.T @ np.linalg.inv(gamma_y) @ phiFull) @ gamma_x
mu_xgy = gamma_xgy @ (phiFull.T @ np.linalg.inv(gamma_ygx) @ radNoisy + np.linalg.inv(gamma_x) @ mu_x)

if ABC == 'A':
    np.save('JayanthRicardo/Dec22/radTrueVeg30Aqua70.npy', radNoisy)
    np.save('JayanthRicardo/Dec22/mu_xgy_linModelVeg30Aqua70.npy', mu_xgy)
    np.save('JayanthRicardo/Dec22/gamma_xgy_linModelVeg30Aqua70.npy', gamma_xgy)
    np.save('JayanthRicardo/Dec22/mu_xgy_IsofitVeg30Aqua70.npy', state_est)
    np.save('JayanthRicardo/Dec22/gamma_xgy_IsofitVeg30Aqua70.npy', S_hat)
elif ABC == 'B':
    np.save('JayanthRicardo/Dec22/radTrueVeg70Aqua30.npy', radNoisy)
    np.save('JayanthRicardo/Dec22/mu_xgy_linModelVeg70Aqua30.npy', mu_xgy)
    np.save('JayanthRicardo/Dec22/gamma_xgy_linModelVeg70Aqua30.npy', gamma_xgy)
    np.save('JayanthRicardo/Dec22/mu_xgy_IsofitVeg70Aqua30.npy', state_est)
    np.save('JayanthRicardo/Dec22/gamma_xgy_IsofitVeg70Aqua30.npy', S_hat)
elif ABC == 'C':
    np.save('JayanthRicardo/Dec22/radTrueVeg50Aqua50.npy', radNoisy)
    np.save('JayanthRicardo/Dec22/mu_xgy_linModelVeg50Aqua50.npy', mu_xgy)
    np.save('JayanthRicardo/Dec22/gamma_xgy_linModelVeg50Aqua50.npy', gamma_xgy)
    np.save('JayanthRicardo/Dec22/mu_xgy_IsofitVeg50Aqua50.npy', state_est)
    np.save('JayanthRicardo/Dec22/gamma_xgy_IsofitVeg50Aqua50.npy', S_hat)


radLin = phi.dot((truth - meanX) / np.sqrt(varX)) * np.sqrt(varY) + meanY
plt.figure(11)
setup.plotbands(radiance[:425],'b', label='Isofit')
setup.plotbands(radLin[:425], 'm',label='Linear Model')
plt.xlabel('Wavelength')
plt.ylabel('Radiance')
if ABC == 'A':
    plt.title('Veg30Aqua70')
elif ABC == 'B':
    plt.title('Veg70Aqua30')  
elif ABC == 'C':
    plt.title('Veg50Aqua50')    
plt.grid()
plt.legend()

plt.figure(12)
setup.plotbands(truth[:425], 'k', label='True Reflectance')
setup.plotbands(state_est[:425],'b', label='Isofit')
setup.plotbands(mu_xgy[:425], 'm',label='Linear Model')
plt.xlabel('Wavelength')
plt.ylabel('Reflectance')
if ABC == 'A':
    plt.title('Veg30Aqua70')
elif ABC == 'B':
    plt.title('Veg70Aqua30')  
elif ABC == 'C':
    plt.title('Veg50Aqua50')  
plt.grid()
plt.legend()


plt.figure(13)
plt.contourf(gamma_xgy)
plt.colorbar()





plt.show()



