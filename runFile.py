import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmc import MCMC


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
x0 = setup.mu_x
yobs = setup.radNoisy
rank = 427
sd = 2.4 ** 2 / min(rank,427)

m.initValue(x0=x0, yobs=yobs, sd=sd, Nsamp=100000, burn=10000, project=True, nr=rank)
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
m.diagnostics(indSet)

plt.show()
