import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

from fileProcessing import FileProcessing
from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmcIsofit import MCMCIsofit


##### CONFIG #####
Nsamp = 6000
burn = 1000
init = 'midMAPtruth'
rank = 100
LIS = True
mcmcfolder = 'G14'
##### CONFIG #####

## SETUP ##
# wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T

# wvl, wv, wvr = np.loadtxt('setup/data/wavelengths.txt').T
# wv = wv * 1000
# wvRaw, refRaw, refnoise = np.loadtxt('setup/data/beckmanlawn/insitu.txt').T
# ref = np.interp(wv, wvRaw, refRaw)
# datamatfile = 'setup/data/beckmanlawn/ang20171108t184227_data_v2p11_BeckmanLawn.mat'
# datamatfile = ''

f = FileProcessing()
f.loadWavelength('setup/data/177/ang20140612t215931_data_dump.mat')
f.loadReflectance('setup/data/177/insitu.txt')
f.loadRadiance('setup/data/177/ang20140612t215931_data_dump.mat')
wv, ref, radiance = f.getFiles()

# f.splitFile(filename='MCMC_x.npy', output='../results/')
# f.mergeFile(inputdir='../results/')


atm = [0.1, 2.5]
# setup = Setup(wv, ref, atm, mcmcdir=mcmcfolder, datamatfile=datamatfile)
setup = Setup(wv, ref, atm, radiance, mcmcdir=mcmcfolder)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

## MCMC #
if init == 'MAP':
    x0 = setup.isofitMuPos
elif init == 'truth':
    x0 = setup.truth
elif init == 'linpos':
    x0, gammapos = a.posterior(setup.radNoisy)
elif init == 'midMAPtruth':
    x0 = 0.5 * (setup.isofitMuPos + setup.truth)
mcmcfolder = mcmcfolder + '_init' + init + '_rank' + str(rank)

m = MCMCIsofit(setup, a, Nsamp, burn, x0, 'AM')
m.initMCMC(LIS=LIS, rank=rank) # specify LIS parameters

start_time = time.time()
m.runAM()
m.calcMeanCov()
setup.plotPosterior(MCMCmean, MCMCcov)

## MCMC Diagnostics ##
# m.mcmcPlots()




'''
# indSet = [30,40,90,100,150,160,250,260, m.nx-2, m.nx-1]
# m.diagnostics(MCMCmean, MCMCcov, indSet)
# np.savetxt(setup.mcmcDir + 'runtime.txt', np.array([time.time() - start_time]))
'''

