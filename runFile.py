import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmcIsofit import MCMCIsofit
from plots import Plots


##### CONFIG #####
Nsamp = 6000000
burn = 1000000
init = 'truth'
rank = 100
LIS = True
mcmcfolder = 'E1'
##### CONFIG #####

## SETUP ##
wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T
atm = [0.5, 2.5]
setup = Setup(wv, ref, atm, mcmcdir=mcmcfolder)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

## MCMC #
if init == 'MAP':
    x0 = setup.isofitMuPos
elif init == 'truth':
    x0 = setup.truth
mcmcfolder = mcmcfolder + '_init' + init + '_rank' + str(rank)

m = MCMCIsofit(setup, a, Nsamp, burn, x0, 'AM')
m.initMCMC(LIS=LIS, rank=rank) # specify LIS parameters


start_time = time.time()
m.runAM()
MCMCmean, MCMCcov = m.calcMeanCov()
setup.plotPosterior(m.linMuPos, m.linGammaPos, MCMCmean, MCMCcov)

## MCMC Diagnostics ##
indSet = [30,40,90,100,150,160,250,260,425,426]
m.diagnostics(MCMCmean, MCMCcov, indSet)
np.savetxt(setup.mcmcDir + 'runtime.txt', np.array([time.time() - start_time]))



