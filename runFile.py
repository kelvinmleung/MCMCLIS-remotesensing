import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmcIsofit import MCMCIsofit


##### CONFIG #####
Nsamp = 6000000
burn = 1000000
init = 'MAP'
rank = 175
mcmcfolder = 'C7'
##### CONFIG #####

## SETUP ##
wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T
atm = [0.05, 2.5]
setup = Setup(wv, ref, atm, mcmcdir=mcmcfolder)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

## MCMC #
start_time = time.time()

if init == 'MAP':
    x0 = setup.isofitMuPos
elif init == 'truth':
    x0 = setup.truth

mcmcfolder = mcmcfolder + '_init' + init + '_rank' + str(rank)

# constrain = False
# if constrain == True:
#     mcmcfolder = mcmcfolder + '_constrained'

m = MCMCIsofit(setup, a, Nsamp, burn, x0, 'AM')
m.initMCMC(LIS=True, rank=rank) # specify LIS parameters
m.runAM()
MCMCmean, MCMCcov = m.calcMeanCov()
setup.plotPosterior(m.linMuPos, m.linGammaPos, MCMCmean, MCMCcov)

## MCMC Diagnostics ##
indSet = [30,40,90,100,150,160,250,260,425,426]
m.diagnostics(MCMCmean, MCMCcov, indSet)
np.savetxt(setup.mcmcDir + 'runtime.txt', np.array([time.time() - start_time]))



# plt.show()

