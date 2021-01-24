import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmc import MCMC

'''
### Notes ###
r175, 3e6 Samples, start truth, atm=[5, 2.5]
'''

## PARAMETERS TO CHANGE ##
# also need to change x0 of MCMC #
rank = 175
sd = 2.4 ** 2 / min(rank,427)
Nsamp = 3000000
burn = 30000

## SETUP ##
wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T
atm = [0.5,2.5]
setup = Setup(wv, ref, atm)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

## MCMC ##
m = MCMC(setup, a)
# mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radNoisy)
x0 = np.zeros(427)
#x0[:425] = setup.isofitMuPos[:425]
x0[:425] = setup.truth[:425]
x0[425:] = [5,2.5]
# x0 = setup.truth

m.initValue(x0=x0, yobs=setup.radNoisy, sd=sd, Nsamp=Nsamp, burn=burn, project=True, nr=rank)
m.runMCMC(alg='adaptive')


