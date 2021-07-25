import numpy as np
import matplotlib.pyplot as plt
from isofitSetup import Setup
from fileProcessing import FileProcessing


f = FileProcessing(setupDir='setup/ang20140612')
f.loadWavelength('data/wavelengths.txt')
f.loadReflectance('data/mars/insitu.txt')
f.loadRadiance('data/mars/ang20140612t215931_data_dump.mat')
f.loadConfig('config/config_inversion.json')
wv, ref, radiance, config = f.getFiles()
setup = Setup(wv, ref, radiance, config)



'''
# get already generated results for David - June 8, 2021
setup.genStartPoints()

truths = np.load('x0isofit/truths.npy')
radiance = np.load('x0isofit/radiance.npy')
'''

'''
### ADDED in inverse.py on line 347
# x0atm = np.load('x0isofit/atmSample.npy')
# x0[425:] = x0atm

### ADDED in inverse.py AFTER xopt = least_squares(err, x0, jac=jac, **self.least_squares_params)
# np.save('/Users/KelvinLeung/Documents/JPLproject/MCMCLIS-remotesensing/x0isofit/nfevTmp.npy', xopt.nfev)
# np.save('/Users/KelvinLeung/Documents/JPLproject/MCMCLIS-remotesensing/x0isofit/statusTmp.npy', xopt.status)

### ADDED in isofitSetup.py after self.truth = np.concatenate((ref, atm))  
# np.save('x0isofit/atmSample.npy', [atm[0], atm[1]]) # set for Isofit initialization

'''
# setup.testIsofitStartPt(1000)

# x = np.load('posAOD_x0isofit.npy')
# y = np.load('posH2O_x0isofit.npy')

setup.testIsofitStartPtPlot()

plt.show()
