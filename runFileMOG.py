
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
# atm = [0.05,1.75]

# Xtrain = np.load('../results/Regression/samples/MOG/X_train.npy')
# Ytrain = np.load('../results/Regression/samples/MOG/Y_train.npy')

wv, aqua = np.loadtxt('../Dec2020/JayanthRicardo/aquatic_spectrum_resampled.txt').T
wv, veg = np.loadtxt('../Dec2020/JayanthRicardo/vegetation_spectrum_resampled.txt').T
# phi = np.load('../results/Regression/MOG/phi.npy')

ABC = 'E'

if ABC == 'A':
    truthRef = 0.3 * veg + 0.7 * aqua # prior 6
    name = 'Veg30Aqua70.npy'
elif ABC == 'B':
    truthRef = 0.7 * veg + 0.3 * aqua # prior 6
    name = 'Veg70Aqua30.npy'
elif ABC == 'C':
    truthRef = 0.5 * veg + 0.5 * aqua # prior 6
    name = 'Veg50Aqua50.npy'
elif ABC == 'D':
    truthRef = veg # prior 6
    name = 'Veg.npy'
elif ABC == 'E':
    truthRef = aqua # prior 3
    name = 'Aqua.npy'

## SETUP ##
setup = Setup(wv, truthRef, atm)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

truth = np.concatenate((truthRef, np.array(atm)))

radiance = setup.radiance
radNoisy = setup.radNoisy
mu_x = setup.mu_x
gamma_x = setup.gamma_x
isofitMuPos = setup.isofitMuPos
isofitGammaPos = setup.isofitGammaPos

meanX = r.meanX
varX = r.varX
meanY = r.meanY
varY = r.varY

gamma_ygx = a.gamma_ygx
phi = a.phi_tilde
phiFull = a.phi

gamma_y = phiFull @ gamma_x @ phiFull.T + gamma_ygx
gamma_xgy = (np.identity(a.nx) - gamma_x @ phiFull.T @ np.linalg.inv(gamma_y) @ phiFull) @ gamma_x
mu_xgy = gamma_xgy @ (phiFull.T @ np.linalg.inv(gamma_ygx) @ radNoisy + np.linalg.inv(gamma_x) @ mu_x)

direc = '../results/MOG/'

np.save(direc + 'radTrue' + name, radNoisy)
np.save(direc + 'mu_xgy_linModel' + name, mu_xgy)
np.save(direc + 'gamma_xgy_linModel' + name, gamma_xgy)
np.save(direc + 'mu_xgy_Isofit' + name, isofitMuPos)
np.save(direc + 'gamma_xgy_Isofit' + name, isofitGammaPos)
np.save(direc + 'mu_x_Isofit' + name, mu_x)
np.save(direc + 'gamma_x_Isofit' + name, gamma_x)

np.save(direc + 'meanX.npy', meanX)
np.save(direc + 'varX.npy', varX)
np.save(direc + 'meanY.npy', meanY)
np.save(direc + 'varY.npy', varY)
np.save(direc + 'phi.npy', phi)
np.save(direc + 'gamma_ygx.npy', gamma_ygx)

radLin = phi.dot((truth - meanX) / np.sqrt(varX)) * np.sqrt(varY) + meanY
plt.figure(11)
plt.plot(wv, radiance[:425],'b', label='Isofit')
plt.plot(wv, radLin[:425], 'm',label='Linear Model')
plt.xlabel('Wavelength')
plt.ylabel('Radiance')
plt.title(name)
plt.grid()
plt.legend()

plt.figure(12)
setup.plotbands(truth[:425], 'k', label='True Reflectance')
setup.plotbands(isofitMuPos[:425],'b', label='Isofit')
setup.plotbands(mu_xgy[:425], 'm',label='Linear Model')
plt.xlabel('Wavelength')
plt.ylabel('Reflectance')
plt.title(name)
plt.grid()
plt.legend()

# plt.figure(13)
# plt.contourf(gamma_xgy)
# plt.colorbar()

plt.show()



