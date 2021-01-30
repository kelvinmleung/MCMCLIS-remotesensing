import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmcIsofit import MCMCIsofit
### Notes ###

## SETUP ##
wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T
atm = [0.5,2.5]
setup = Setup(wv, ref, atm)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

## MCMC ##
x0 = setup.mu_x
Nsamp = 100000
burn = 10000

m = MCMCIsofit(setup, a, Nsamp, burn, x0)
m.initMCMC(LIS=False, rank=10) # specify LIS parameters
m.runAM()

# compare posterior mean
# mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radNoisy)
MCMCmean, MCMCcov = m.calcMeanCov()
# setup.plotPosterior(mu_xgyLin, gamma_xgyLin, MCMCmean, MCMCcov)

## MCMC Diagnostics ##
indSet = [30,40,90,100,150,160,250,260,425,426]
indSet = [0,1,2,3,4,5,6,7,8,9]
m.diagnostics(MCMCmean, MCMCcov, indSet)

plt.show()