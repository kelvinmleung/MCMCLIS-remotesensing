import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmcIsofit import MCMCIsofit


##### CONFIG #####
Nsamp = 3000
burn = 1000
init = 'isofit'
rank = 100
constrain = True
mcmcfolder = 'b1'
mcmcfolder = mcmcfolder + '_init' + init + '_rank' + str(rank)
if constrain == True:
    mcmcfolder = mcmcfolder + '_constrained'
##### CONFIG #####

## SETUP ##
wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T
atm = [0.5, 2.5]
setup = Setup(wv, ref, atm, mcmcdir=mcmcfolder)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

## MCMC #
if init == 'isofit':
    x0 = setup.isofitMuPos
elif init == 'truth':
    x0 = setup.truth


m = MCMCIsofit(setup, a, Nsamp, burn, x0, 'AM')
m.initMCMC(LIS=True, rank=rank, constrain=constrain) # specify LIS parameters
m.runAM()
MCMCmean, MCMCcov = m.calcMeanCov()
setup.plotPosterior(m.linMuPos, m.linGammaPos, MCMCmean, MCMCcov)

## MCMC Diagnostics ##
indSet = [30,40,90,100,150,160,250,260,425,426]
m.diagnostics(MCMCmean, MCMCcov, indSet)

# zip the plots
listFiles = []
listFiles = listFiles + ['reflMean.png', 'atmMean.png', 'reflVar.png', 'atmVar.png', ' errorRelCov.png', 'trace.png', 'autocorr.png', 'logpos.png', 'acceptance.png']
numPairs = int(len(indSet) / 2)
for i in range(numPairs):
    listFiles = listFiles + ['2D_' + str(indSet[2*i]) + '-' + str(indSet[2*i+1]) + '.png']
subprocess.call(['tar', '-zcvf', mcmcfolder + '.tgz', listFiles])

# plt.show()

