import sys, os, json
import numpy as np
import scipy as s
import matplotlib.pyplot as plt

from plots import PlotFromFile

mcmcfolder = 'G11'
p = PlotFromFile(mcmcfolder)
# p.plotPosterior()
p.plotError()
# p.plot2Dmarginal()
# p.kdcontour(indX=425, indY=426)
# p.diagnostics(indSet=[10,20,50,100,150,160,250,260])
p.quantDiagnostic()
plt.show()

