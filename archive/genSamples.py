import numpy as np

class GenerateSamples:
    '''
    Contains functions to generate training and test samples
    from isofit.
    '''
    def __init__(self, setup):
        self.sampleDir = setup.sampleDir

        self.setup = setup
        self.fm = setup.fm
        self.geom = setup.geom 
        self.noisecov = setup.noisecov
        
    def genTrainingSamples(self, Nsamp):
        fm = self.fm
        geom = self.geom
        mu_x, gamma_x = self.setup.getPrior()

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
        
    def genTestSamples(self, N):

        fm = self.fm
        geom = self.geom

        mu_x, gamma_x = self.setup.getPrior()

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

    def genY(self):
        # use this if we are given X and want to get radiances Y
        X_train = np.load(self.sampleDir + 'X_train.npy')
        X_test = np.load(self.sampleDir + 'X_test.npy')

        Ntrain = X_train.shape[0]
        Ntest = X_test.shape[0]
        nx = X_train.shape[1]
        ny = nx - 2

        Y_train = np.zeros([Ntrain,ny])
        Y_test = np.zeros([Ntest,ny])

        for i in range(Ntrain):
            meas = self.fm.calc_rdn(X_train[i,:], self.geom)
            eps_samp = np.random.multivariate_normal(np.zeros(ny), self.noisecov)
            Y_train[i,:] = meas + eps_samp
            if (i+1) % 100 == 0:
                print('Training: Iteration ', i+1)

        for i in range(Ntest):
            meas = self.fm.calc_rdn(X_test[i,:], self.geom)
            eps_samp = np.random.multivariate_normal(np.zeros(ny), self.noisecov)
            Y_test[i,:] = meas + eps_samp
            if (i+1) % 100 == 0:
                print('Test: Iteration ', i+1)

        np.save(self.sampleDir + 'Y_train.npy', Y_train)
        np.save(self.sampleDir + 'Y_test.npy', Y_test)


