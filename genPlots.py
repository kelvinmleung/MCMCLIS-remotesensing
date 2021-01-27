import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmc import MCMC
### Notes ###
'''
1. Run on cluster
2. Run transferFromSSH.py
3. Move files to /results/MCMC
4. Copy the setup from runFileSLURM.py to this file
4. Run genPlots.py
'''

## SETUP ##
wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T
atm = [0.5,2.5]
setup = Setup(wv, ref, atm)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

## MCMC ##
m = MCMC(setup, a)

mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radNoisy)
x0 = np.zeros(427)
x0[:425] = setup.truth[:425]
# x0[:425] = mu_xgyLin[:425]
x0[425:] = [5,2.5]
# x0 = setup.truth

yobs = setup.radNoisy
rank = 175
sd = 2.4 ** 2 / min(rank,427)
Nsamp = 300000
burn = 30000

m.initValue(x0=x0, yobs=yobs, sd=sd, Nsamp=Nsamp, burn=burn, project=True, nr=rank)
MCMCmean, MCMCcov = m.calcMeanCov()

# compare posterior mean
mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radNoisy)
setup.plotPosterior(mu_xgyLin, gamma_xgyLin, MCMCmean, MCMCcov)

## MCMC Diagnostics ##
indSet = [10,20,50,100,150,160,250,260,425,426]
m.diagnostics(MCMCmean, MCMCcov, indSet, calcAC=True)



plt.show()
