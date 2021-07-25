import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmcIsofit import MCMCIsofit


## SETUP ##
wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T
atm = [0.05, 2.5]
setup = Setup(wv, ref, atm, mcmcdir=mcmcfolder)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)


a.plotcontour(setup.gamma_x, 'Prior Covariance')
a.plotcontour(setup.noisecov, 'Observation Noise Covariance')

a.comparePosParam(gamma_xgy, 300)
a.comparePosData(gamma_xgy, 300)


plt.show()


