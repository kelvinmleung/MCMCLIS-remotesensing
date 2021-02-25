import sys, os, json
import numpy as np
import scipy as s
import matplotlib.pyplot as plt

from plotFromFile import PlotFromFile


mcmcfolder = 'B8'
p = PlotFromFile(mcmcfolder)

# p.plotRegression()
# p.plot2Dmarginal()
# p.plotposmean()
# p.plotposvar()
# p.plot2ac()
p.plotkdcontour(indX=90, indY=100)
plt.title('LIS - 100 Dimensions')


plt.show()