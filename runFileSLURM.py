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
atm = [0.5, 2.5]
setup = Setup(wv, ref, atm)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

## MCMC ##
x0 = setup.isofitMuPos
Nsamp = 2000000
burn = 200000

m = MCMCIsofit(setup, a, Nsamp, burn, x0)
m.initMCMC(LIS=False, rank=427) # specify LIS parameters
m.runAM()

MCMCmean, MCMCcov = m.calcMeanCov()

## MCMC Diagnostics ##
mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radNoisy)
setup.plotPosterior(mu_xgyLin, gamma_xgyLin, MCMCmean, MCMCcov)
indSet = [30,40,90,100,150,160,250,260,425,426]
m.diagnostics(MCMCmean, MCMCcov, indSet)



