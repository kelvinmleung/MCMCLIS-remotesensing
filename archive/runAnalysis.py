import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis

## SETUP ##
wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T
atm = [0.5, 2.5]
setup = Setup(wv, ref, atm)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

eigval, eigvec = a.eigLIS()
a.eigPlots(eigval, eigvec, rank=427, title='LIS')

a.eigLISdata()

plt.show()