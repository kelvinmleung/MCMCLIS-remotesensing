import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

from fileProcessing import FileProcessing
from isofitSetup import Setup
from plots import PlotFromFile

f = FileProcessing()
f.loadWavelength('setup/data/177/ang20140612t215931_data_dump.mat')
f.loadReflectance('setup/data/177/insitu.txt')
f.loadRadiance('setup/data/177/ang20140612t215931_data_dump.mat')
f.loadConfig('setup/config/config_inversion_JPL.json')
wv, ref, radiance, config = f.getFiles()

mcmcfolder = 'TestDimension'
atm = [0.1, 2.5]
setup = Setup(wv, ref, atm, radiance, config, mcmcdir=mcmcfolder)
setup.saveConfig()


p = PlotFromFile(mcmcfolder)
plt.figure()
p.plotbands(setup.truth, 'b-', label='Truth')
p.plotbands(setup.isofitMuPos, 'r-', label='Isofit')
plt.legend()

plt.show()

