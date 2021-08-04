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
init = 'MAP'
rank = 100
LIS = True
mcmcfolder = 'H01'
thinning = 20
setupDir = 'ang20170228'
##### CONFIG #####

f = FileProcessing(setupDir='setup/' + setupDir)
f.loadWavelength('data/wavelengths.txt')
f.loadReflectance('data/beckmanlawn/insitu.txt')
f.loadRadiance('data/beckmanlawn/ang20171108t184227_data_v2p11_BeckmanLawn.mat')
f.loadConfig('config/config_inversion.json')

# f.loadWavelength('data/wavelengths.txt')
# f.loadReflectance('data/306/insitu.txt')
# f.loadRadiance('data/306/ang20140612t215931_data_dump.mat')
# f.loadConfig('config/config_inversion.json')
wv, ref, radiance, config = f.getFiles()


setup = Setup(wv, ref, radiance, config, mcmcdir=mcmcfolder, setupDir=setupDir)
g = GenerateSamples(setup)

# surf_mu, surf_gamma = f.loadSurfModel('data/surface.mat')
# atm_mu = setup.mu_x[432:]
# atm_gamma = setup.gamma_x[432:, 432:]
# atm_bounds = np.array([[0.001, 0.5],[1.3100563704967498, 1.586606174707413]])

# g.genTrainTest(surf_mu, surf_gamma, atm_mu, atm_gamma, atm_bounds, f, 'train', NperPrior=3000)
# g.genTrainTest(surf_mu, surf_gamma, atm_mu, atm_gamma, atm_bounds, f, 'test', NperPrior=1000)

r = Regression(setup)

# for i in [10,20,100,120,250,260,400]:
#     print('yElem = ', i)
#     r.tuneLasso(params=[1e-4,1e-3,1e-2,1e-1], yElem=i, plot=False)
# r.fullLasso(params=np.ones(432)*1e-3)
# r.plotFullLasso()
# plt.show()

a = Analysis(setup, r)

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








