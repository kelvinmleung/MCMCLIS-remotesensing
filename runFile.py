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
atm = [0.05, 1.75]
setup = Setup(wv, ref, atm)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

## MCMC ##
# x0 = np.zeros(427)
# x0[:425] = setup.isofitMuPos[:425]
# x0[425:] = [0.05, 1.75]
x0 = setup.mu_x
Nsamp = 200000
burn = 20000

m = MCMCIsofit(setup, a, Nsamp, burn, x0)
m.initMCMC(LIS=False, rank=427) # specify LIS parameters
m.runAM()
MCMCmean, MCMCcov = m.calcMeanCov()

# compare posterior mean
mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radNoisy)
setup.plotPosterior(mu_xgyLin, gamma_xgyLin, MCMCmean, MCMCcov)

## MCMC Diagnostics ##
indSet = [30,40,90,100,150,160,250,260,425,426]
# indSet = [0,1,2,3,4,5,6,7,8,9]
m.diagnostics(MCMCmean, MCMCcov, indSet)

plt.show()