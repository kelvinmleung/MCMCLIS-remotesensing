import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmc import MCMC


## SETUP ##
atm = [0.1,2]
setup = Setup(atm)

## GENERATE SAMPLES ##
g = GenerateSamples(setup)
# g.genTrainingSamples(10000)
# g.genTestSamples(2000)

## REGRESSION ##
r = Regression(setup, g)
# r.fullLasso([1e-2] * 425)
# r.plotFullLasso()

## ANALYSIS ##
a = Analysis(setup, r)

## MCMC ##
m = MCMC(setup, a)
x0 = setup.truth
rank = 427
sd = 2.4 ** 2 / min(rank,427)

m.initValue(x0=x0, yobs=setup.radiance, sd=sd, Nsamp=1000000, burn=10000, project=False, nr=rank)
# m.runMCMC(alg='adaptive')
MCMCmean, MCMCcov = m.calcMeanCov()
m.plotMCMCmean(MCMCmean, fig=1)

# compare posterior mean
mu_x, gamma_x = setup.getPrior()
mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radiance)
mu_xgyLinNoise, gamma_xgyLinNoise = a.posterior_noise(yobs=setup.radiance)
isofitMuPos = setup.isofitMuPos
setup.plotPosMean(isofitMuPos, mu_xgyLin,  mu_xgyLinNoise, MCMCmean)

## MCMC Diagnostics ##
indSet = [10,20,50,100,150,160,250,260,425,426]
m.diagnostics(indSet)

#diagnostic = np.load(setup.mcmcDir + 'NoLIS_ClipNoise1e-7/diagnostic.npy')
'''
diagnostic = np.load(setup.mcmcDir + 'NoLIS_NoClipNoise/diagnostic.npy')

diagnostic = diagnostic[:,900000:]
meanDiag = np.mean(diagnostic,1)
medianDiag = np.mean(diagnostic,1)
stddev = np.sqrt(np.var(diagnostic,1))
maxDiag = np.max(diagnostic,1)
minDiag = np.min(diagnostic,1)
plt.figure(1)

plt.semilogy(setup.wavelengths, maxDiag,'k.', label='Max')
#plt.semilogy(setup.wavelengths, minDiag,'k.', label='Min')
#plt.semilogy(setup.wavelengths[::5], meanDiag[::5],'b.',label='Mean')
#plt.semilogy(setup.wavelengths, medianDiag,'r.',label='Median')
plt.grid()
plt.axvline(x=setup.wavelengths[185],color='r')
plt.axvline(x=setup.wavelengths[214],color='r')
plt.axvline(x=setup.wavelengths[281],color='r')
plt.axvline(x=setup.wavelengths[314],color='r')
plt.axvline(x=setup.wavelengths[414],color='r')

plt.title('Likelihood Diagnostic')
plt.xlabel('Wavelength')
plt.legend()
'''
plt.show()
