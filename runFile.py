import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmc import MCMC


## SETUP ##
atm = [0.3,2]
setup = Setup(atm)

## GENERATE SAMPLES ##
indPr = 6
g = GenerateSamples(setup)
# g.genTrainingSamples(10000, indPr)
# g.genTestSamples(2000, indPr)

## REGRESSION ##
r = Regression(setup, g)
r.fullLasso([1e-2] * 425)
r.plotFullLasso

## ANALYSIS ##
#a = Analysis(setup, r)

# ## MCMC ##
# m = MCMC(setup, a)
# x0 = setup.truth
# rank = 150
# sd = 2.4 ** 2 / min(rank,350)

# m.initValue(x0=x0, yobs=setup.radiance, sd=sd, Nsamp=50000, project=True, nr=rank)
# #m.runMCMC(alg='adaptive')
# MCMCmean, MCMCcov = m.calcMeanCov(N=8000)
# m.plotMCMCmean(MCMCmean, fig=1, mcmcType='pos')

# # compare posterior mean
# mu_x, gamma_x = setup.getPrior(6)
# mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radiance)
# mu_xgyLinNoise, gamma_xgyLinNoise = a.posterior_noise(yobs=setup.radiance)
# isofitMuPos = setup.isofitMuPos
# setup.plotPosMean(isofitMuPos, mu_xgyLin,  mu_xgyLinNoise, MCMCmean)

# ## MCMC Diagnostics ##
# indSet = [10,20,50,100,150,160,250,260,425,426]
# m.diagnostics(indSet)
# # m.twoDimVisual(indX=10, indY=20, t0=0)
# # m.twoDimVisual(indX=50, indY=100, t0=0)
# # m.twoDimVisual(indX=150, indY=160, t0=0)
# # m.twoDimVisual(indX=250, indY=260, t0=0)
# # m.twoDimVisual(indX=425, indY=426, t0=0)


plt.show()
