#import sys, os, json
#import scipy as s
import numpy as np
#import matplotlib.pyplot as plt

'''
from scipy.io import loadmat
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale, StandardScaler
'''
'''
sys.path.insert(0, '../isofit/')

import isofit
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.inversion.inverse import Inversion
from isofit.configs.configs import Config  
'''
class GenerateSamples:
    '''
    Contains functions to generate training and test samples
    from isofit.
    '''
    # deleted the fixed_atm in this version, and changed getPrior outputs from 4 to 2
    # also deleted the first genTestSamples function
    def __init__(self, setup):

        self.wavelengths, self.reflectance = np.loadtxt('setup/data/petunia/petunia_reflectance.txt').T

        self.setup = setup
        self.fm = self.setup.fm
        self.geom = self.setup.geom 

        self.truth = self.setup.truth
        self.radiance = self.setup.radiance
        
    def genTrainingSamples(self, Nsamp, idx_pr=0):
        fm = self.fm
        geom = self.geom
        #idx_pr = 2 #for vegetation
        mu_x, gamma_x = self.setup.getPrior(idx_pr)

        nx = gamma_x.shape[0]
        ny = nx - 2 #gamma_priorsurf.shape[0]
        mu_ygx = np.zeros(ny)
        x_samp = np.zeros([Nsamp, nx])
        y_samp = np.zeros([Nsamp, ny])

        for i in range(Nsamp):
            while x_samp[i,425] <= 0 or x_samp[i,425] > 1 or x_samp[i,426] < 1 or x_samp[i,426] > 4:
                # CHANGE THIS TO FASTER MULTIVARIATE NORMAL SAMPLING
                x_samp[i,:] = np.random.multivariate_normal(mu_x, gamma_x)
            meas = fm.calc_meas(x_samp[i,:], geom)
            gamma_ygx = fm.instrument.Sy(meas, geom)

            eps_samp = np.random.multivariate_normal(mu_ygx, gamma_ygx)
            y_samp[i][:] = meas + eps_samp
            print('Sampling: Iteration ', i+1)

        np.save('results/samples/X_train.npy', x_samp)
        np.save('results/samples/Y_train.npy', y_samp)
        
    def genTestSamples(self, N, idx_pr=0):

        fm = self.fm
        geom = self.geom

        #idx_pr = 2 #for vegetation
        mu_x, gamma_x = self.setup.getPrior(idx_pr)

        nx = gamma_x.shape[0]
        ny = nx - 2
        mu_ygx = np.zeros(ny)
        x_samp = np.zeros([N, nx])
        y_samp = np.zeros([N, ny])

        for i in range(N):
            while x_samp[i,425] <= 0 or x_samp[i,425] > 1 or x_samp[i,426] < 1 or x_samp[i,426] > 4:
                x_samp[i,:] = np.random.multivariate_normal(mu_x, gamma_x)

            meas = fm.calc_meas(x_samp[i,:], geom)
            gamma_ygx = fm.instrument.Sy(meas, geom)

            eps_samp = np.random.multivariate_normal(mu_ygx, gamma_ygx)
            y_samp[i][:] = meas + eps_samp
            print('Sampling: Iteration ', i+1)

        np.save('results/samples/X_test.npy', x_samp)
        np.save('results/samples/Y_test.npy', y_samp)
    '''
    def fwdModel(self):

        print('Forward Model Setup...')
        with open('setup/config/config_inversion.json', 'r') as f:
            config = json.load(f)
        geom = Geometry()
        fm_config = Config(config)
        fm = ForwardModel(fm_config)
        print('Setup Finished.')

        return fm, geom

    def invModel(self, fm, geom):
        
        #fm, geom = self.fwdModel()
        
        print('Running Inverse Model...')

        inversion_settings = {"implementation": {
        "mode": "inversion",
        "inversion": {
        "windows": [[380.0, 1300.0], [1450, 1780.0], [1950.0, 2450.0]]}}}
            
        inverse_config = Config(inversion_settings)
        iv = Inversion(inverse_config, fm)

        radiance = fm.calc_rdn(self.truth, geom)

        state_trajectory = iv.invert(radiance, geom)
        state_est = state_trajectory[-1]

        rfl_est, rdn_est, path_est, S_hat, K, G = iv.forward_uncertainty(state_est, radiance, geom)
        #A = s.matmul(G,K)

        print('Inversion finished.')

        return state_est, S_hat#, rdn_est, path_est
    '''