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
Nsamp = 6000
burn = 2000
init = 'MAP'
rank = 100
LIS = True
mcmcfolder = 'H11'
thinning = 20
setupDir = 'ang20140628'
##### CONFIG #####

f = FileProcessing(setupDir='setup/' + setupDir)
# f.loadWavelength('data/wavelengths.txt')
# f.loadReflectance('data/beckmanlawn/insitu.txt')
# f.loadRadiance('data/beckmanlawn/ang20171108t184227_data_v2p11_BeckmanLawn.mat')
# f.loadConfig('config/config_inversion.json')
f.loadWavelength('data/wavelengths.txt')
f.loadReflectance('data/177/insitu.txt')
f.loadRadiance('data/177/ang20140612t215931_data_dump.mat')
f.loadConfig('config/config_inversion.json')
wv, ref, radiance, config = f.getFiles()

setup = Setup(wv, ref, radiance, config, mcmcdir=mcmcfolder, setupDir=setupDir)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

linPosMu, linPosGamma = a.posterior(setup.radiance)
plt.figure()
plt.plot(setup.wavelengths[setup.bands], setup.truth[setup.bands], 'k.', label='Truth')
plt.plot(setup.wavelengths[setup.bands], setup.isofitMuPos[setup.bands], 'r.', label='Isofit')
plt.plot(setup.wavelengths[setup.bands], linPosMu[setup.bands], 'b.', label='Linear Posterior')
plt.show()

print('ATM Parameters')
print('Isofit:', setup.isofitMuPos[-2:])
print('Linear:', linPosMu[-2:])


# plt.figure()
# plt.plot(setup.wavelengths[setup.bands], setup.truth[setup.bands], 'k.', label='Truth')
# plt.plot(setup.wavelengths[setup.bands], setup.isofitMuPos[setup.bands], 'r.', label='Isofit')
# plt.show()











