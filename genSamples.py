import numpy as np

class GenerateSamples:
    '''
    Contains functions to generate training and test samples
    from isofit.
    '''
    def __init__(self, setup):
        ### ADD INDPRIOR AS INPUT AND TRANSFER TO ANALYSIS>PY
        self.sampleDir = '../results/Regression/samples/'

        self.setup = setup
        self.fm = self.setup.fm
        self.geom = self.setup.geom 
        
    def genTrainingSamples(self, Nsamp, idx_pr=6):
        fm = self.fm
        geom = self.geom
        mu_x, gamma_x = self.setup.getPrior(idx_pr)

        nx = gamma_x.shape[0]
        ny = nx - 2
        mu_ygx = np.zeros(ny)
        x_samp = np.zeros([Nsamp, nx])
        y_samp = np.zeros([Nsamp, ny])

        cholGammaX = np.linalg.cholesky(gamma_x)

        for i in range(Nsamp):
            while x_samp[i,425] <= 0 or x_samp[i,425] > 1 or x_samp[i,426] < 1 or x_samp[i,426] > 4:
                z = np.random.normal(0,1,size=nx)
                x_samp[i,:] = mu_x + cholGammaX @ z
            meas = fm.calc_meas(x_samp[i,:], geom)
            gamma_ygx = fm.Seps(x_samp[i,:], meas, geom)

            eps_samp = np.random.multivariate_normal(mu_ygx, gamma_ygx)
            y_samp[i][:] = meas + eps_samp

            if (i+1) % 100 == 0:
                print('Sampling: Iteration ', i+1)

        np.save(self.sampleDir + 'X_train.npy', x_samp)
        np.save(self.sampleDir + 'Y_train.npy', y_samp)
        
    def genTestSamples(self, N, idx_pr=6):

        fm = self.fm
        geom = self.geom

        mu_x, gamma_x = self.setup.getPrior(idx_pr)

        nx = gamma_x.shape[0]
        ny = nx - 2
        mu_ygx = np.zeros(ny)
        x_samp = np.zeros([N, nx])
        y_samp = np.zeros([N, ny])

        cholGammaX = np.linalg.cholesky(gamma_x)

        for i in range(N):
            while x_samp[i,425] <= 0 or x_samp[i,425] > 1 or x_samp[i,426] < 1 or x_samp[i,426] > 4:
                z = np.random.normal(0,1,size=nx)
                x_samp[i,:] = mu_x + cholGammaX @ z

            meas = fm.calc_meas(x_samp[i,:], geom)
            gamma_ygx = fm.Seps(x_samp[i,:], meas, geom)

            eps_samp = np.random.multivariate_normal(mu_ygx, gamma_ygx)
            y_samp[i][:] = meas + eps_samp
            print('Sampling: Iteration ', i+1)

        np.save(self.sampleDir + 'X_test.npy', x_samp)
        np.save(self.sampleDir + 'Y_test.npy', y_samp)