import sys, os, json
import numpy as np
import scipy as s
from scipy.io import loadmat
import matplotlib.pyplot as plt

sys.path.insert(0, '../isofit/')

import isofit
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.inversion.inverse import Inversion
from isofit.configs.configs import Config  
    
class Setup:
    '''
    Contains functions to generate training and test samples
    from isofit.
    '''
    # deleted the fixed_atm in this version, and changed getPrior outputs from 4 to 2
    # also deleted the first genTestSamples function
    def __init__(self, atm):
        # atm is the atmospheric parameters

        self.wavelengths, self.reflectance = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T

        self.fm, self.geom = self.fwdModel()
        self.truth = np.concatenate((self.reflectance, atm)) 

        rad = self.fm.calc_rdn(self.truth, self.geom)
        self.noisecov = self.fm.Seps(self.truth, rad, self.geom)
        eps = np.random.multivariate_normal(np.zeros(len(rad)), self.noisecov)
        self.radiance = rad
        self.radNoisy = rad + eps
        
        self.isofitMuPos, self.isofitGammaPos = self.invModel(self.radiance)

        wl = self.wavelengths
        bands = []
        for i in range(wl.size):
            if (wl[i] > 380 and wl[i] < 1300) or (wl[i] > 1450 and wl[i] < 1780) or (wl[i] > 1950 and wl[i] < 2450):
                bands = bands + [i]
        self.bands = bands
        self.bandsX = bands + [425,426]
    
    def getPrior(self, idx_pr=0):
        fm = self.fm
        # Get prior mean and covariance
        
        surfmat = loadmat('setup/data/surface.mat')
        wl = surfmat['wl'][0]
        refwl = np.squeeze(surfmat['refwl'])
        idx_ref = [np.argmin(abs(wl-w)) for w in np.squeeze(refwl)]
        idx_ref = np.array(idx_ref)
        refnorm = np.linalg.norm(self.reflectance[idx_ref])

        mu_priorsurf = fm.surface.components[idx_pr][0] * refnorm
        mu_priorRT = fm.RT.xa()
        mu_priorinst = fm.instrument.xa()
        mu_x = np.concatenate((mu_priorsurf, mu_priorRT, mu_priorinst), axis=0)
        
        gamma_priorsurf = fm.surface.components[idx_pr][1] * (refnorm ** 2)
        gamma_priorRT = fm.RT.Sa()[:, :]
        gamma_priorinst = fm.instrument.Sa()[:, :]
        gamma_x = s.linalg.block_diag(gamma_priorsurf, gamma_priorRT, gamma_priorinst)

        return mu_x, gamma_x

    def fwdModel(self):

        print('Forward Model Setup...')
        with open('setup/config/config_inversion.json', 'r') as f:
            config = json.load(f)
        geom = Geometry()
        fm_config = Config(config)
        fm = ForwardModel(fm_config)
        print('Setup Finished.')

        return fm, geom

    def invModel(self, radiance):

        fm = self.fm
        geom = self.geom
                
        print('Running Inverse Model...')

        inversion_settings = {"implementation": {
        "mode": "inversion",
        "inversion": {
        "windows": [[380.0, 1300.0], [1450, 1780.0], [1950.0, 2450.0]]}}}

        inverse_config = Config(inversion_settings)
        iv = Inversion(inverse_config, fm)

        state_trajectory = iv.invert(radiance, geom)
        state_est = state_trajectory[-1]

        rfl_est, rdn_est, path_est, S_hat, K, G = iv.forward_uncertainty(state_est, radiance, geom)
        #A = s.matmul(G,K)

        print('Inversion finished.')

        return state_est, S_hat#, rdn_est, path_est

        

    def plotbands(self, y, linestyle, linewidth=1, label='', axis='normal'):
        wl = self.wavelengths
        if axis == 'normal':
            plt.plot(wl[1:185], y[1:185], linestyle, linewidth=linewidth, label=label)
            plt.plot(wl[215:281], y[215:281], linestyle, linewidth=linewidth)
            plt.plot(wl[315:414], y[315:414], linestyle, linewidth=linewidth)
        elif axis == 'semilogy':
            plt.semilogy(wl[1:185], y[1:185], linestyle, linewidth=linewidth, label=label)
            plt.semilogy(wl[215:281], y[215:281], linestyle, linewidth=linewidth)
            plt.semilogy(wl[315:414], y[315:414], linestyle, linewidth=linewidth)

    def plotCov(self, gamma):
        plt.figure()
        X_plot = np.arange(1,426,1)
        Y_plot = np.arange(1,426,1)
        X_plot, Y_plot = np.meshgrid(X_plot, Y_plot)
        plt.contourf(X_plot,Y_plot,gamma)
        plt.title('Covariance')
        plt.axis('equal')
        plt.colorbar()

    def plotPosMean(self, isofitMuPos, mu_xgyLin,  mu_xgyLinNoise, MCMCmean):
        mu_x, gamma_x = self.getPrior(6)

        plt.figure(64)
        self.plotbands(self.truth[:425], 'b.',label='True Reflectance')
        #self.plotbands(mu_x[:425], 'r.',label='Prior')
        self.plotbands(isofitMuPos[:425],'k.', label='Isofit Posterior')
        self.plotbands(mu_xgyLin[:425], 'm.',label='Linear Posterior')
        #self.plotbands(mu_xgyLinNoise[:425], 'g.',label='Linear - Noise Covariance')
        self.plotbands(MCMCmean[:425], 'c.',label='MCMC Posterior')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.grid()
        plt.legend()

        isofitError = abs(isofitMuPos[:425] - self.truth[:425]) / abs(self.truth[:425])
        linError = abs(mu_xgyLin[:425] - self.truth[:425]) / abs(self.truth[:425])
        mcmcError = abs(MCMCmean[:425] - self.truth[:425]) / abs(self.truth[:425])
        isofitMCMC =abs(isofitMuPos[:425] - MCMCmean[:425]) / abs(self.truth[:425])
        plt.figure(65)
        self.plotbands(isofitError,'k.', label='Isofit Posterior',axis='semilogy')
        self.plotbands(linError, 'm.',label='Linear Posterior',axis='semilogy')
        self.plotbands(mcmcError, 'c.',label='MCMC Posterior',axis='semilogy')
        self.plotbands(isofitMCMC, 'b.',label='MCMC/Isofit',axis='semilogy')
        plt.xlabel('Wavelength')
        plt.ylabel('Relative Error')
        plt.grid()
        plt.legend()

        plt.figure(66)
        plt.plot(self.truth[425], self.truth[426], 'bo',label='True Reflectance')
        plt.plot(mu_x[425], mu_x[426], 'r.',label='Prior')
        plt.plot(isofitMuPos[425],isofitMuPos[426],'k.', label='Isofit Posterior')
        #plt.plot(mu_xgyLin[425], mu_xgyLin[426],'mx',label='Linear Posterior')
        #plt.plot(mu_xgyLinNoise[425],mu_xgyLinNoise[426], 'gx',label='Linear - Noise Covariance')
        plt.plot(MCMCmean[425], MCMCmean[426], 'cx',label='MCMC Posterior')
        plt.xlabel('AOT550')
        plt.ylabel('H2OSTR')
        plt.grid()
        plt.legend()
