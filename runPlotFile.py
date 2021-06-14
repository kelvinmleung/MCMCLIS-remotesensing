import sys, os, json
import numpy as np
import scipy as s
import matplotlib.pyplot as plt

from plotFromFile import PlotFromFile


mcmcfolder = 'C7'
p = PlotFromFile(mcmcfolder)

# p.plotRegression()
p.plot2Dmarginal()
p.plotposmean()
p.plotposvar()

# p.plot2ac(indset=[20,50,90,120,150,200,230,250,400,410,425,426])

p.plotkdcontour(indX=90, indY=100)
# plt.title('r = 175')

# p.plotCompareRank()

# p.plotPosSparsity(1e-4)
# p.plotPosSparsity(1e-5)
# p.plotPosSparsity(1e-6)
# p.plotPosCovRow()


plt.show()