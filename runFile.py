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
init = 'MAP'
rank = 100
LIS = True
mcmcfolder = 'H_test'
thinning = 20
setupDir = 'ang20140612'#'ang20170228'
##### CONFIG #####

f = FileProcessing(setupDir='setup/' + setupDir)
# f.loadWavelength('data/wavelengths.txt')
# f.loadReflectance('data/beckmanlawn/insitu.txt')
# f.loadRadiance('data/beckmanlawn/ang20171108t184227_data_v2p11_BeckmanLawn.mat')
# f.loadConfig('config/config_inversion.json')


f.loadWavelength('data/wavelengths.txt')
f.loadReflectance('data/177/insitu.txt')
f.loadRadiance('data/177/ang20140612t215931_data_dump.mat')
f.loadConfig('config/config_inversion.json')
wv, ref, radiance, config = f.getFiles()

setup = Setup(wv, ref, radiance, config, mcmcdir=mcmcfolder, setupDir=setupDir)
g = GenerateSamples(setup)

# g.genTrainTest(surf_mu, surf_gamma, atm_mu, atm_gamma, atm_bounds, f, 'train', NperPrior=6000)
# g.genTrainTest(surf_mu, surf_gamma, atm_mu, atm_gamma, atm_bounds, f, 'test', NperPrior=2000)

# X_train = np.load('../results/Regression/samples/ang20140612/X_train.npy')
# X_test = np.load('../results/Regression/samples/ang20140612/X_test.npy')
# g.genY(X_train, X_test)


r = Regression(setup, fixatm=False)

# for i in [10,20,100,120,250,260,400]:
#     print('yElem = ', i)
#     r.tuneLasso(params=[1e-4,1e-3,1e-2,1e-1], yElem=i, plot=False)
# r.fullLasso(params=np.ones(432)*1e-3)
# r.plotFullLasso()
# plt.show()

a = Analysis(setup, r)

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
m.initMCMC(LIS=LIS, rank=rank, constrain=False, fixatm=False) # specify LIS parameters

start_time = time.time()
m.runAM()
np.savetxt(setup.mcmcDir + 'runtime.txt', np.array([time.time() - start_time]))

