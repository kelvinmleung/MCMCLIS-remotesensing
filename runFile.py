import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmcIsofit import MCMCIsofit
### Notes ###
'''
LIS rank 175.
Try fixed and unfixed atm parameters

'''

## SETUP ##
wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T
atm = [0.5, 2.5]
setup = Setup(wv, ref, atm)

g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

## MCMC ##
# x0 = setup.isofitMuPos
x0 = setup.truth
Nsamp = 2000000
burn = int(0.1 * Nsamp)
rank = 100

m = MCMCIsofit(setup, a, Nsamp, burn, x0, 'AM')
m.initMCMC(LIS=True, rank=rank) # specify LIS parameters
m.runAM()
MCMCmean, MCMCcov = m.calcMeanCov()

# compare posterior mean
setup.plotPosterior(m.linMuPos, m.linGammaPos, MCMCmean, MCMCcov)

## MCMC Diagnostics ##
indSet = [30,40,90,100,150,160,250,260,425,426]
m.diagnostics(MCMCmean, MCMCcov, indSet)


plt.show()

