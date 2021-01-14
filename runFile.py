import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmc import MCMC
### Notes ###
'''
Start with x0 at truth for r = 427
Want to test using a larger initial proposal so it can explore more regions. 
Use linear posterior covariance as initial proposal (with no sd multiplied to it)
 
 -> to initValue, add self.propcov = self.linpos # * sd 
'''


## SETUP ##
wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T
atm = [0.5,2.5]
setup = Setup(wv, ref, atm)

## GENERATE SAMPLES ##
g = GenerateSamples(setup)
# g.genTrainingSamples(10000)
# g.genTestSamples(2000)
# g.genY()

## REGRESSION ##
r = Regression(setup)
# r.fullLasso([1e-2] * 425)
# .plotFullLasso()


## ANALYSIS ##
a = Analysis(setup, r)

## MCMC ##
m = MCMC(setup, a)

x0 = np.zeros(427)
x0[:425] = setup.mu_x[:425]
x0[425:] = [5,2.5]

yobs = setup.radNoisy
rank = 427
sd = 2.4 ** 2 / min(rank,427)

m.initValue(x0=x0, yobs=yobs, sd=sd, Nsamp=500000, burn=50000, project=True, nr=rank)
m.runMCMC(alg='adaptive')
MCMCmean, MCMCcov = m.calcMeanCov()
m.plotMCMCmean(MCMCmean, fig=1)

# compare posterior mean
mu_x = setup.mu_x
gamma_x = setup.gamma_x
mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radiance)
mu_xgyLinNoise, gamma_xgyLinNoise = a.posterior_noise(yobs=setup.radiance)
setup.plotPosMean(mu_xgyLin,  mu_xgyLinNoise, MCMCmean)

## MCMC Diagnostics ##
indSet = [10,20,50,100,150,160,250,260,425,426]
m.diagnostics(indSet, calcAC=True)

plt.show()
