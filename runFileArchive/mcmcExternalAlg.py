
import numpy as np
import scipy as s
import matplotlib.pyplot as plt

from pymcmcstat.MCMC import MCMC
from pymcmcstat.plotting.utilities import generate_ellipse

from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmcIsofit import MCMCIsofit

# run AM in pymcmcstat environment
# prior sampling for sanity check
# https://github.com/prmiles/pymcmcstat/blob/master/tutorials/specifying_sample_variables/specifying_sample_variables.ipynb
# https://github.com/prmiles/pymcmcstat/blob/master/tutorials/banana/Banana.ipynb

wv, ref = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T
atm = [0.5, 2.5]
setup = Setup(wv, ref, atm)
g = GenerateSamples(setup)
r = Regression(setup)
a = Analysis(setup, r)

x0 = setup.mu_x
Nsamp = 500000
burn = int(0.1*Nsamp)

# mu = setup.mu_x[:10]
# gamma = setup.gamma_x[:10,:][:,:10]
rank = 400
mu = np.zeros(rank)
gamma = np.identity(rank)

m = MCMCIsofit(setup, a, Nsamp, burn, x0)
# m.initMCMC(LIS=False, rank=427)
m.initPriorSampling(rank=rank)


class Prior_Parameters:
    def __init__(self, mu, gamma, npar=427):
        self.mu = mu
        self.gamma = gamma
        self.invGamma = np.linalg.inv(gamma)
        self.npar = npar

def target(theta, data):
    udobj = data.user_defined_object[0]
    mu = udobj.mu
    gamma = udobj.gamma
    invGamma = udobj.invGamma
    npar = udobj.npar
    x = np.array([theta])
    x = np.squeeze(x)

    density = normDensity(x, mu, invGamma)
    return density

def normDensity(x, mu, invGamma):
    # logprior = -1/2 * ((x-mu) @ invGamma @ (x-mu).T)
    logprior = (x-mu) @ invGamma @ (x-mu).T

    return logprior
    # return np.exp(logprior)





npar = mu.shape[0]  
initCov = gamma * (2.4 ** 2) / npar
udobj = Prior_Parameters(mu, gamma, npar) # user defined object

mcstat = MCMC()
mcstat.data.add_data_set(np.zeros(1), np.zeros(1),
                         user_defined_object=udobj)
# Add model parameters
for ii in range(npar):
    mcstat.parameters.add_model_parameter(
        name=str('$x_{}$'.format(ii + 1)),
        theta0 = mu[ii])

mcstat.simulation_options.define_simulation_options(
    nsimu=Nsamp,
    updatesigma=False,
    method='am',
    printint=500,
    qcov=initCov) # adaptint=500,

mcstat.model_settings.define_model_settings(sos_function=target)

mcstat.run_simulation()

# Extract results
results = mcstat.simulation_results.results
chain = results['chain']
s2chain = results['s2chain']
names = results['names']

# define burnin
burnin = burn
# display chain statistics
mcstat.chainstats(chain[burnin:, :], results)

np.save(setup.mcmcDir + 'MCMC_x.npy', chain.T)
# np.savetxt(setup.mcmcDir + 'MCMC_x.txt', chain.T)

MCMCmean, MCMCcov = m.calcMeanCov()

# compare posterior mean
# mu_xgyLin, gamma_xgyLin = a.posterior(yobs=setup.radNoisy)
# setup.plotPosterior(mu_xgyLin, gamma_xgyLin, MCMCmean, MCMCcov)

## MCMC Diagnostics ##
indSet = []
for i in range(10):
    indSet = indSet + [int(rank/10 * i)]
m.diagnostics(MCMCmean, MCMCcov, indSet)

plt.show()