import numpy as np
import matplotlib.pyplot as plt

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmcIsofit import MCMCIsofit


## SETUP ##
wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T
atm = [0.05, 2.5]
setup = Setup(wv, ref, atm)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

## AOD COMPARISON
# atm = [0.1, 2.5]
# setup1 = Setup(wv, ref, atm)
# atm = [0.3, 2.5]
# setup2 = Setup(wv, ref, atm)
# atm = [0.5, 2.5]
# setup3 = Setup(wv, ref, atm)

# plt.figure()
# setup.plotbands(ref, 'k', linewidth=3, label='Truth', axis='normal')
# setup.plotbands(setup1.isofitMuPos, 'b', linewidth=1.5, label='AOD = 0.1', axis='normal')
# setup.plotbands(setup2.isofitMuPos, 'r', linewidth=1.5, label='AOD = 0.3', axis='normal')
# setup.plotbands(setup3.isofitMuPos, 'm', linewidth=1.5, label='AOD = 0.5', axis='normal')
# plt.title('Isofit Retrievals with Varying AOD Parameter')
# plt.xlabel('Wavelength [nm]')
# plt.ylabel('Reflectance')
# plt.legend()


# a.plotcontour(setup.gamma_x, 'Prior Covariance')
# a.plotcontour(setup.noisecov, 'Observation Noise Covariance')

## EIGENVALUE PLOTS
# eigvalPCA, eigvecPCA = a.eigPCA()
# eigvalPCAdata, eigvecPCAdata = a.eigPCAdata()
# eigvalLIS, eigvecLIS = a.eigLIS()
# a.plotEig(eigvalPCA, eigvecPCA, title='PCA Parameter Space')
# a.plotEig(eigvalPCAdata, eigvecPCAdata, title='PCA Data Space')
# a.plotEig(eigvalLIS, eigvecLIS, title='LIS')


## BAYES RISK AND FORSTNER
# a.comparePosParam(427)
# a.comparePosData(427)

dims1 = range(5,251,5)
cut1 = len(dims1)

dims2 = range(5,401,5)
cut2 = len(dims2)

gamma_errorPCAparam = np.load(a.analysisDir + 'gamma_errorPCAparam.npy')
gamma_errorLISparam = np.load(a.analysisDir + 'gamma_errorLISparam.npy')
mu_errorPCAparam = np.load(a.analysisDir + 'mu_errorPCAparam.npy')
mu_errorLISparam = np.load(a.analysisDir + 'mu_errorLISparam.npy')

gamma_errorPCAdata = np.load(a.analysisDir + 'gamma_errorPCAdata.npy')
gamma_errorLISdata = np.load(a.analysisDir + 'gamma_errorLISdata.npy')
mu_errorPCAdata = np.load(a.analysisDir + 'mu_errorPCAdata.npy')
mu_errorLISdata = np.load(a.analysisDir + 'mu_errorLISdata.npy')

plt.figure()
plt.semilogy(dims1, gamma_errorPCAparam[:cut1], 'b.', label='PCA')
plt.semilogy(dims1, gamma_errorLISparam[:cut1], 'r.', label='LIS')
plt.xlabel('Dimension of Subspace')
plt.ylabel('Forstner Distance')
plt.title('Error in Posterior Covariance - Parameter Space')
plt.legend()

plt.figure()
plt.semilogy(dims1, mu_errorPCAparam[:cut1], 'b.', label='PCA')
plt.semilogy(dims1, mu_errorLISparam[:cut1], 'r.', label='LIS')
plt.xlabel('Dimension of Subspace')
plt.ylabel('Bayes Risk')
plt.title('Error in Posterior Mean - Parameter Space')
plt.legend()

plt.figure()
plt.semilogy(dims2, gamma_errorPCAdata[:cut2], 'b.', label='PCA')
plt.semilogy(dims2, gamma_errorLISdata[:cut2], 'r.', label='LIS')
plt.xlabel('Dimension of Subspace')
plt.ylabel('Forstner Distance')
plt.title('Error in Posterior Covariance - Data Space')
plt.legend()

plt.figure()
plt.semilogy(dims2, mu_errorPCAdata[:cut2], 'b.', label='PCA')
plt.semilogy(dims2, mu_errorLISdata[:cut2], 'r.', label='LIS')
plt.xlabel('Dimension of Subspace')
plt.ylabel('Bayes Risk')
plt.title('Error in Posterior Mean - Data Space')
plt.legend()


plt.show()


