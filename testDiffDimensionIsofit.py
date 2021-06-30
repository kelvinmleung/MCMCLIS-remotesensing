import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

from fileProcessing import FileProcessing
from isofitSetup import Setup
from plots import PlotFromFile
from fileProcessing import FileProcessing
from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmcIsofit import MCMCIsofit

f = FileProcessing()
f.loadWavelength('setup/data/177/ang20140612t215931_data_dump.mat')
f.loadReflectance('setup/data/177/insitu.txt')
f.loadRadiance('setup/data/177/ang20140612t215931_data_dump.mat')
wv, ref, radiance = f.getFiles()


mcmcfolder = 'TestDimension'
atm = [0.1, 2.5]
setup = Setup(wv, ref, atm, radiance, mcmcdir=mcmcfolder)
g = GenerateSamples(setup)
# r = Regression(setup)
a = Analysis(setup, r)

Nsamp = 6000000
burn = 2000000
init = 'truth'
rank = 100
LIS = True
thinning = 20
x0 = setup.isofitMuPos
m = MCMCIsofit(setup, a, Nsamp, burn, x0, 'AM', thinning=thinning)
m.initMCMC(LIS=LIS, rank=rank) # specify LIS parameters
m.saveConfig()

p = PlotFromFile(mcmcfolder)
plt.figure()
p.plotbands(setup.truth, 'b-', label='Truth')
p.plotbands(setup.isofitMuPos, 'r-', label='Isofit')

plt.show()

