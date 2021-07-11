import sys, os, json
import numpy as np
import scipy as s
import matplotlib.pyplot as plt

from plots import PlotFromFile


# mean, cov = m.calcMeanCov()
# setup.plotPosterior(mean, cov)

## MCMC Diagnostics ##
# m.mcmcPlots()

# indSet = [30,40,90,100,150,160,250,260, m.nx-2, m.nx-1]
# m.diagnostics(MCMCmean, MCMCcov, indSet)

mcmcfolder = 'G11'
p = PlotFromFile(mcmcfolder)
p.plotPosterior()
p.plot2Dmarginal()

p.kdcontour(indX=432, indY=433)
p.diagnostics(indSet=[10,20,50,100,150,160,250,260])

plt.show()

