
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


atm = [0.5,2.5]


# Xtrain = np.load('../results/Regression/samples/MOG/X_train.npy')
# Ytrain = np.load('../results/Regression/samples/MOG/Y_train.npy')


wv, aqua = np.loadtxt('../Dec2020/JayanthRicardo/aquatic_spectrum_resampled.txt').T
wv, veg = np.loadtxt('../Dec2020/JayanthRicardo/vegetation_spectrum_resampled.txt').T
# phi = np.load('../results/Regression/MOG/phi.npy')

# meanX = np.mean(Xtrain,0)
# varX = np.var(Xtrain,0)
# meanY = np.mean(Ytrain,0)
# varY = np.var(Ytrain,0)

# scaleX = (Xtrain - meanX) / np.sqrt(varX)
# scaleY = (Ytrain - meanY) / np.sqrt(varY)

ABC = 'B'


if ABC == 'A':
    truthRef = 0.3 * veg + 0.7 * aqua # prior 6
elif ABC == 'B':
    truthRef = 0.7 * veg + 0.3 * aqua # prior 6
elif ABC == 'C':
    truthRef = 0.5 * veg + 0.5 * aqua # prior 6

# truthRef = veg #6
# truthRef = aqua #3 

## SETUP ##
setup = Setup(wv, truthRef, atm)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

meanX = r.meanX
varX = r.varX
meanY = r.meanY
varY = r.varY


radiance = setup.radiance
radNoisy = setup.radNoisy

state_est = setup.isofitMuPos
S_hat = setup.isofitGammaPos

truth = np.concatenate((truthRef, np.array(atm)))
mu_x = setup.mu_x
gamma_x = setup.gamma_x

gamma_ygx = a.gamma_ygx
phi = a.phi_tilde
phiFull = a.phi

# eps = np.random.multivariate_normal(np.zeros(len(radiance)), gamma_ygx)
# radNoisyNew = radiance + eps

gamma_y = phiFull @ gamma_x @ phiFull.T + gamma_ygx
gamma_xgy = (np.identity(a.nx) - gamma_x @ phiFull.T @ np.linalg.inv(gamma_y) @ phiFull) @ gamma_x
mu_xgyNOISY = gamma_xgy @ (phiFull.T @ np.linalg.inv(gamma_ygx) @ radNoisy + np.linalg.inv(gamma_x) @ mu_x)
mu_xgy = gamma_xgy @ (phiFull.T @ np.linalg.inv(gamma_ygx) @ radiance + np.linalg.inv(gamma_x) @ mu_x)
'''
if ABC == 'A':
    np.save('../Dec2020/JayanthRicardo/Dec29/radTrueVeg30Aqua70.npy', radNoisy)
    np.save('../Dec2020/JayanthRicardo/Dec29/mu_xgy_linModelVeg30Aqua70.npy', mu_xgy)
    np.save('../Dec2020/JayanthRicardo/Dec29/gamma_xgy_linModelVeg30Aqua70.npy', gamma_xgy)
    np.save('../Dec2020/JayanthRicardo/Dec29/mu_xgy_IsofitVeg30Aqua70.npy', state_est)
    np.save('../Dec2020/JayanthRicardo/Dec29/gamma_xgy_IsofitVeg30Aqua70.npy', S_hat)
    np.save('../Dec2020/JayanthRicardo/Dec29/mu_x_IsofitVeg30Aqua70.npy', mu_x)
    np.save('../Dec2020/JayanthRicardo/Dec29/gamma_x_IsofitVeg30Aqua70.npy', gamma_x)
elif ABC == 'B':
    np.save('../Dec2020/JayanthRicardo/Dec29/radTrueVeg70Aqua30.npy', radNoisy)
    np.save('../Dec2020/JayanthRicardo/Dec29/mu_xgy_linModelVeg70Aqua30.npy', mu_xgy)
    np.save('../Dec2020/JayanthRicardo/Dec29/gamma_xgy_linModelVeg70Aqua30.npy', gamma_xgy)
    np.save('../Dec2020/JayanthRicardo/Dec29/mu_x_IsofitVeg70Aqua30.npy', mu_x)
    np.save('../Dec2020/JayanthRicardo/Dec29/gamma_x_IsofitVeg70Aqua30.npy', gamma_x)
    np.save('../Dec2020/JayanthRicardo/Dec29/mu_xgy_IsofitVeg70Aqua30.npy', state_est)
    np.save('../Dec2020/JayanthRicardo/Dec29/gamma_xgy_IsofitVeg70Aqua30.npy', S_hat)
elif ABC == 'C':
    np.save('../Dec2020/JayanthRicardo/Dec29/radTrueVeg50Aqua50.npy', radNoisy)
    np.save('../Dec2020/JayanthRicardo/Dec29/mu_xgy_linModelVeg50Aqua50.npy', mu_xgy)
    np.save('../Dec2020/JayanthRicardo/Dec29/gamma_xgy_linModelVeg50Aqua50.npy', gamma_xgy)
    np.save('../Dec2020/JayanthRicardo/Dec29/mu_x_IsofitVeg50Aqua50.npy', mu_x)
    np.save('../Dec2020/JayanthRicardo/Dec29/gamma_x_IsofitVeg50Aqua50.npy', gamma_x)
    np.save('../Dec2020/JayanthRicardo/Dec29/mu_xgy_IsofitVeg50Aqua50.npy', state_est)
    np.save('../Dec2020/JayanthRicardo/Dec29/gamma_xgy_IsofitVeg50Aqua50.npy', S_hat)
'''

radLin = phi.dot((truth - meanX) / np.sqrt(varX)) * np.sqrt(varY) + meanY
plt.figure(11)
plt.plot(wv, radiance[:425],'b', label='Isofit')
plt.plot(wv, radNoisy[:425],'r', label='Isofit (Plus Noise)')
#plt.plot(wv, radNoisyNew[:425],'r', label='Isofit (Plus Noise GammaYGX)')
#setup.plotbands(radLin[:425], 'm',label='Linear Model')
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
setup.plotbands(mu_xgyNOISY[:425], 'r',label='Linear Model Noisy')
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



