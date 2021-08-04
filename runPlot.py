import sys, os, json
import numpy as np
import scipy as s
import matplotlib.pyplot as plt

from plots import PlotFromFile

###### MAKE THE 2D MARGINAL BUT IN KD CONTOUR FORM ######

mcmcfolder = 'H01'
setupDir = 'setup/ang20170228/' #'setup/ang20140612/'
p = PlotFromFile(mcmcfolder, setupDir)
p.plotPosterior()
p.plotError()
p.plot2Dmarginal()
p.kdcontour(indX=425, indY=426)
p.diagnostics(indSet=[10,20,50,100,150,160,250,260])
p.quantDiagnostic()
plt.show()



# mcmcDir = '../results/MCMC/' 
# isofitG115 = np.load(mcmcDir + 'G115_2/isofitMuPos.npy')
# isofitG11 = np.load(mcmcDir + 'G11_2/isofitMuPos.npy')
# truthG11 = np.load(mcmcDir + 'G11_2/truth.npy')
# # isofitG119 = np.load(mcmcDir + 'G119/isofitMuPos.npy')
# # isofitG116 = np.load(mcmcDir + 'G116/isofitMuPos.npy')
# # isofitG112 = np.load(mcmcDir + 'G112/isofitMuPos.npy')

# print(isofitG115[:10])
# print(isofitG11[:10])
# # print(isofitG119[:10])
# # print(isofitG116[:10])
# # print(isofitG112[:10])

# plt.figure()
# plt.plot(truthG11)
# plt.plot(isofitG115)
# plt.plot(isofitG11)
# # plt.plot(isofitG119)
# # plt.plot(isofitG116)
# # plt.plot(isofitG112)
# plt.show()

