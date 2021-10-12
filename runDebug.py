# Aug 7, 2021
# Ran the MCMC with a new linear model for the real radiance data
# MCMC results a lot worse than Isofit optimization
# To check: linear model, isofit inversion results

import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

from fileProcessing import FileProcessing
from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmcIsofit import MCMCIsofit


##### CONFIG #####
# Nsamp = 6000
burn = 2000
# init = 'MAP'
# rank = 100
# LIS = True
mcmcfolder = 'H34'
thinning = 20
setupDir = 'ang20140612'#'ang20170228'#
##### CONFIG #####

f = FileProcessing(setupDir='setup/' + setupDir)
# f.loadWavelength('data/wavelengths.txt')
# f.loadReflectance('data/beckmanlawn/insitu.txt')
# f.loadRadiance('data/beckmanlawn/ang20171108t184227_data_v2p11_BeckmanLawn.mat')
# f.loadConfig('config/config_inversion.json')
f.loadWavelength('data/wavelengths.txt')
f.loadReflectance('data/dark/insitu.txt')
f.loadRadiance('data/dark/ang20140612t215931_data_dump.mat')
f.loadConfig('config/config_inversion.json')
wv, ref, radiance, config = f.getFiles()

# radiance = 0 # USE SIMULATED DATA
setup = Setup(wv, ref, radiance, config, mcmcdir=mcmcfolder, setupDir=setupDir)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)


# noisevar = np.diag(setup.noisecov)
# Dinv = np.linalg.inv(np.diag(np.sqrt(noisevar)))
# print(np.median(setup.radianceSim / np.sqrt(noisevar)))

# plt.figure()
# plt.semilogy(noisevar)
# plt.title('Config SNR 500')

# plt.figure()
# plt.semilogy(setup.radianceSim / np.sqrt(noisevar))

# corrMatrix = Dinv @ setup.noisecov @ Dinv

# plt.figure()
# plt.contourf(np.log10(corrMatrix + np.ones(corrMatrix.shape) * 1e-72),levels=[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0])#-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0]) #
# plt.colorbar()
# plt.title('Config SNR 500 - Corr Mat')

# plt.show()



'''
x_scaled = (setup.truth - r.meanX) / np.sqrt(r.varX)
y_lasso = np.diag(np.sqrt(r.varY)) @ a.phi_tilde @ x_scaled + r.meanY
plt.figure()
plt.plot(setup.wavelengths, setup.radianceSim, 'b', label='Isofit')
plt.plot(setup.wavelengths, y_lasso, 'r', label='Linear Model')
plt.plot(setup.wavelengths, radiance, 'g', label='Real Data')
plt.legend()
plt.title('Simulated Radiance')
plt.show()
'''

# X_train = np.diag(np.sqrt(r.varX)) @ r.X_train.T + np.outer(r.meanX, np.ones(25000))
# X_train = X_train.T



# for j in range(10):
#     ind = np.random.randint(0,25000)
#     X_sample = X_train[ind,:]

#     X_train_new = np.outer(X_sample, np.ones(1000)).T # np.zeros([1000,len(X_sample)])
#     Y_train_new = np.zeros([X_train_new.shape[0], X_train_new.shape[1]-2])
#     for i in range(1000):
#         atm_aerosol = np.random.uniform(0.001, 0.5, 1)
#         atm_h2o = np.random.uniform(1.3100563704967498, 1.586606174707413, 1)
#         X_train_new[i,432:] = [atm_aerosol, atm_h2o]
#         Y_train_new[i,:] = setup.fm.calc_rdn(X_train_new[i,:], setup.geom)

#     plt.figure()
#     for i in range(1000):
#         plt.plot(Y_train_new[i,:])
#     plt.show()

# plt.figure()

# plt.semilogy(setup.isofitGammaPos[250,:], color='b')
# plt.title('Row of Cov Matrix - Index 250')
# plt.show()

# X_plot = np.arange(1,setup.ny+1,1)
# Y_plot = np.arange(1,setup.ny+1,1)
# X_plot, Y_plot = np.meshgrid(X_plot, Y_plot)
# plt.figure()
# plt.contourf(X_plot,Y_plot,setup.gamma_x[:setup.nx-2,:setup.nx-2])
# plt.title('Prior Covariance')
# plt.axis('equal')
# plt.colorbar()

# print(setup.isofitMuPos[425:])
linPosMu, linPosGamma = a.posterior(setup.radiance)
x_vals = np.load('../results/MCMC/'+mcmcfolder+'/MCMC_x.npy', mmap_mode='r')
x_plot = x_vals[:,int(burn/thinning):]
MCMCmean = np.mean(x_plot, axis=1)

# linPosMu, linPosGamma = a.posterior(setup.radianceSim)
plt.figure()
plt.plot(setup.wavelengths[setup.bands], setup.truth[setup.bands], 'k.', label='Truth')
# plt.plot(setup.wavelengths[setup.bands], setup.mu_x[setup.bands], 'g.', label='Prior')
plt.plot(setup.wavelengths[setup.bands], setup.isofitMuPos[setup.bands], 'r.', label='Isofit')
plt.plot(setup.wavelengths[setup.bands], linPosMu[setup.bands], 'g.', label='Linear Posterior')
plt.plot(setup.wavelengths[setup.bands], MCMCmean[setup.bands], 'b.', label='MCMC')

plt.legend()
plt.ylabel('Reflectance')
plt.title('Retrieval')
plt.show()


# eigval, eigvec = a.eigLIS()
# a.eigPlots(eigval, eigvec, rank=434, title='LIS')
# plt.show()

# print('ATM Parameters')
# print('Isofit:', setup.isofitMuPos[-2:])
# print('Linear:', linPosMu[-2:])


# plt.figure()
# plt.plot(setup.wavelengths[setup.bands], setup.truth[setup.bands], 'k.', label='Truth')
# plt.plot(setup.wavelengths[setup.bands], setup.isofitMuPos[setup.bands], 'r.', label='Isofit')
# plt.plot(setup.wavelegnths[setup.bands], setup.)
# plt.show()


# r.distFromTruth()
# plt.show()










