import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmc import MCMC
### Notes ###
'''
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

mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radiance)
x0 = np.zeros(427)
x0[:425] = setup.truth[:425]
x0[425:] = [5,2.5]
#x0 = setup.truth

yobs = setup.radNoisy
rank = 175
sd = 2.4 ** 2 / min(rank,427)
Nsamp = 5000
burn = 1000

m.initValue(x0=x0, yobs=yobs, sd=sd, Nsamp=Nsamp, burn=burn, project=True, nr=rank)
m.runMCMC(alg='adaptive')
MCMCmean, MCMCcov = m.calcMeanCov()
# m.plotMCMCmean(MCMCmean, fig=1)

# compare posterior mean
mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radiance)
# mu_xgyLinNoise, gamma_xgyLinNoise = a.posterior_noise(yobs=setup.radiance)
setup.plotPosterior(mu_xgyLin,  gamma_xgyLin, MCMCmean, MCMCcov)

## MCMC Diagnostics ##
indSet = [10,20,50,100,150,160,250,260,425,426]
m.diagnostics(indSet, calcAC=True)

plt.show()
