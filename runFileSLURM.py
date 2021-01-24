import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmc import MCMC
### Notes ###
'''
r250, 3e5 Samples, start lin pos refl, atm=[1, 2.5]

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

# mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radNoisy)
x0 = np.zeros(427)
x0[:425] = setup.isofitMuPos[:425]
x0[425:] = [1,2.5]
# x0 = setup.truth

yobs = setup.radNoisy
rank = 250
sd = 2.4 ** 2 / min(rank,427)
Nsamp = 3000
burn = 300

m.initValue(x0=x0, yobs=yobs, sd=sd, Nsamp=Nsamp, burn=burn, project=True, nr=rank)
m.runMCMC(alg='adaptive')
MCMCmean, MCMCcov = m.calcMeanCov()

'''
# compare posterior mean
mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radNoisy)
setup.plotPosterior(mu_xgyLin, gamma_xgyLin, MCMCmean, MCMCcov)

## MCMC Diagnostics ##
indSet = [10,20,50,100,150,160,250,260,425,426]
m.diagnostics(indSet, calcAC=True)

# save figures
# figs = [plt.figure(n) for n in plt.get_fignums()]
# for fig in figs:
#     fig.savefig(str(fig), format='png')

plt.show()
'''
