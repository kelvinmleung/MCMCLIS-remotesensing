import sys, os, json
import numpy as np
import scipy as s
import matplotlib.pyplot as plt

from plots import PlotFromFile


mcmcfolder = 'H33'
setupDir = 'setup/ang20140612/' #'setup/ang20170228/' #
p = PlotFromFile(mcmcfolder, setupDir)
p.plotPosterior()
p.plotError()
p.plot2Dmarginal()
# p.plot2Dcontour()
p.kdcontouratm(indX=432, indY=433)
p.diagnostics(indSet=[20,50,150,160,250,260,400,410])
p.quantDiagnostic()

fig, ax = plt.subplots()
p.twoDimVisual(432,433,ax)
plt.title('2D Marginal - Atmospheric Parameters')

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

