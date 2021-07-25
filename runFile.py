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
burn = 2000
init = 'linpos'
rank = 100
LIS = True
mcmcfolder = 'G11_2'
thinning = 20
##### CONFIG #####

# f = FileProcessing(setupDir='setup/ang20170228')
# f.loadWavelength('data/wavelengths.txt')
# f.loadReflectance('data/beckmanlawn/insitu.txt')
# f.loadRadiance('data/beckmanlawn/ang20171108t184227_data_v2p11_BeckmanLawn.mat')
# f.loadConfig('config/config_inversion.json')
f = FileProcessing(setupDir='setup/ang20140612')
f.loadWavelength('data/wavelengths.txt')
f.loadReflectance('data/306/insitu.txt')
f.loadRadiance('data/306/ang20140612t215931_data_dump.mat')
f.loadConfig('config/config_inversion.json')
wv, ref, radiance, config = f.getFiles()


setup = Setup(wv, ref, radiance, config, mcmcdir=mcmcfolder)
g = GenerateSamples(setup)
# r = Regression(setup)
# a = Analysis(setup, r)

# linPosMu, linPosGamma = a.posterior(setup.radiance)
# plt.figure()
# plt.plot(setup.wavelengths[setup.bands], setup.truth[setup.bands], 'k.', label='Truth')
# plt.plot(setup.wavelengths[setup.bands], setup.isofitMuPos[setup.bands], 'r.', label='Isofit')
# plt.plot(setup.wavelengths[setup.bands], linPosMu[setup.bands], 'b.', label='Linear Posterior')
# plt.show()

# plt.figure()
# plt.plot(setup.wavelengths[setup.bands], setup.truth[setup.bands], 'k.', label='Truth')
# plt.plot(setup.wavelengths[setup.bands], setup.isofitMuPos[setup.bands], 'r.', label='Isofit')
# plt.show()


'''
## MCMC #
if init == 'MAP':
    x0 = setup.isofitMuPos
elif init == 'truth':
    x0 = setup.truth
elif init == 'midMAPtruth':
    x0 = 0.5 * (setup.isofitMuPos + setup.truth)
elif init == 'linpos':
    linMuPos, linGammaPos = a.posterior(setup.radiance)
    x0 = linMuPos

mcmcfolder = mcmcfolder + '_init' + init + '_rank' + str(rank)

m = MCMCIsofit(setup, a, Nsamp, burn, x0, 'AM', thinning=thinning)
m.initMCMC(LIS=LIS, rank=rank) # specify LIS parameters

start_time = time.time()
m.runAM()
np.savetxt(setup.mcmcDir + 'runtime.txt', np.array([time.time() - start_time]))

'''






