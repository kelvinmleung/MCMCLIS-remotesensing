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
# burn = 2000
# init = 'MAP'
# rank = 100
# LIS = True
mcmcfolder = 'H11S_test'
# thinning = 20
setupDir = 'ang20140612'#'ang20170228'#
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

radiance = 0 # USE SIMULATED DATA
setup = Setup(wv, ref, radiance, config, mcmcdir=mcmcfolder, setupDir=setupDir)
g = GenerateSamples(setup)
# r = Regression(setup)
# a = Analysis(setup, r)

# g.genGMM(30000)
# g.addAtm()

X_train = np.load(setup.sampleDir + 'X_train.npy')
X_test = np.load(setup.sampleDir + 'X_test.npy')
g.genY(X_train, X_test)








