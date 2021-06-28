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
Nsamp = 6000000
burn = 2000000
init = 'truth'
rank = 100
LIS = True
mcmcfolder = 'F2'
thinning = 20
##### CONFIG #####

f = FileProcessing()
f.loadWavelength('setup/data/beckmanlawn/ang20171108t184227_data_v2p11_BeckmanLawn.mat')
f.loadReflectance('setup/data/beckmanlawn/insitu.txt')
f.loadRadiance('setup/data/beckmanlawn/ang20171108t184227_data_v2p11_BeckmanLawn.mat')
wv, ref, radiance = f.getFiles()


atm = [0.1, 2.5]
setup = Setup(wv, ref, atm, radiance, mcmcdir=mcmcfolder)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

## MCMC #
if init == 'MAP':
    x0 = setup.isofitMuPos
elif init == 'truth':
    x0 = setup.truth
elif init == 'midMAPtruth':
    x0 = 0.5 * (setup.isofitMuPos + setup.truth)
mcmcfolder = mcmcfolder + '_init' + init + '_rank' + str(rank)

m = MCMCIsofit(setup, a, Nsamp, burn, x0, 'AM', thinning=thinning)
m.initMCMC(LIS=LIS, rank=rank) # specify LIS parameters

start_time = time.time()
m.runAM()
m.saveConfig()
np.savetxt(setup.mcmcDir + 'runtime.txt', np.array([time.time() - start_time]))


