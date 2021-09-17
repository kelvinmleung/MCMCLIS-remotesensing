import numpy as np
from scipy.io import loadmat

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
        # given one prior
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

    def getReflectance(self, f, randInd):
        refl = np.zeros([4,432])
        refl[0,:] = f.loadReflectance('data/177/insitu.txt')
        refl[1,:] = f.loadReflectance('data/306/insitu.txt')
        refl[2,:] = f.loadReflectance('data/mars/insitu.txt')
        refl[3,:] = f.loadReflectance('data/dark/insitu.txt')
        return refl[randInd, :]

    def convertSurfScale(self, mu, gamma, refl):
        # scales the surface prior to a (random) reflectance

        # Get prior mean and covariance
        surfmat = loadmat(self.setup.surfaceFile)
        wl = surfmat['wl'][0]
        refwl = np.squeeze(surfmat['refwl'])
        idx_ref = [np.argmin(abs(wl-w)) for w in np.squeeze(refwl)]
        idx_ref = np.array(idx_ref)
        refnorm = np.linalg.norm(refl[idx_ref])
        # refnorm = np.linalg.norm(self.reflectance[idx_ref])

        # mu_priorsurf = self.fm.surface.components[indPr][0] * refnorm
        # gamma_priorsurf = self.fm.surface.components[indPr][1] * (refnorm ** 2)

        mu_scaled = mu * refnorm
        gamma_scaled = gamma * (refnorm ** 2)

        return mu_scaled, gamma_scaled

    def genTrainTest(self, surf_mu, surf_gamma, atm_mu, atm_gamma, atm_bounds, f, traintest, NperPrior=5000):
        # given a surface model, generate train and test samples using all 8 priors
        # fm = self.fm
        # geom = self.geom
        # mu_x, gamma_x = self.setup.getPrior()

        # numPriors = surf_mu.shape[0]
        numPriors = 4
        Nsamp = NperPrior * numPriors
        ny = surf_mu.shape[1]
        nx = ny + len(atm_mu)

        mu_ygx = np.zeros(ny)
        x_samp = np.zeros([Nsamp, nx])
        y_samp = np.zeros([Nsamp, ny])

        indPr = np.array([2,3,7,2])

        import matplotlib.pyplot as plt

        for i in range(numPriors):
            for j in range(NperPrior):
                k = i * NperPrior + j

                refl = self.getReflectance(f, i)
                # refl = self.getReflectance(f, np.random.randint(0,4))
                surf_mu_scaled, surf_gamma_scaled = self.convertSurfScale(surf_mu, surf_gamma, refl)

                mu_x = np.concatenate((surf_mu_scaled[indPr[i]], atm_mu))
                gamma_x = np.zeros([nx, nx])
                gamma_x[:ny, :ny] = surf_gamma_scaled[[indPr[i]],:,:]
                gamma_x[ny:, ny:] = np.diag(atm_gamma)
                cholGammaX = np.linalg.cholesky(gamma_x)
                while x_samp[k,nx-2] < atm_bounds[0,0] or x_samp[k,nx-2] > atm_bounds[0,1] or x_samp[k,nx-1] < atm_bounds[1,0] or x_samp[k,nx-1] > atm_bounds[1,1]:
                    
                    z = np.random.normal(0,1,size=nx)
                    x_samp[k,:] = abs(mu_x + cholGammaX @ z)
                # print(np.sqrt(np.diag(gamma_x)))
                
                meas = self.fm.calc_meas(x_samp[k,:], self.geom)
                gamma_ygx = self.fm.Seps(x_samp[k,:], meas, self.geom)
                
                eps_samp = np.random.multivariate_normal(mu_ygx, gamma_ygx)
                # print(eps_samp)
                y_samp[k,:] = meas + eps_samp

                # plt.figure(1)
                # plt.plot(self.setup.wavelengths[self.setup.bands], x_samp[k,self.setup.bands])
                # plt.figure(2)
                # plt.plot(self.setup.wavelengths, y_samp[k,:])
                # plt.show()

                if (k+1) % 100 == 0:
                    print('Sampling: Iteration ', k+1)

        if traintest == 'train':
            np.save(self.sampleDir + 'X_train.npy', x_samp)
            np.save(self.sampleDir + 'Y_train.npy', y_samp)
        elif traintest == 'test':
            np.save(self.sampleDir + 'X_test.npy', x_samp)
            np.save(self.sampleDir + 'Y_test.npy', y_samp)

    def genY(self, X_train, X_test):
        # generates Y_train and Y_test given X_train and X_test
        ny = X_train.shape[1] - 2
        numTrain = X_train.shape[0]
        numTest = X_test.shape[0]
        Y_train = np.zeros([numTrain, ny])
        Y_test = np.zeros([numTest, ny])
        mu_ygx = np.zeros(ny)

        for k in range(numTrain):
            meas = self.fm.calc_meas(X_train[k,:], self.geom)
            gamma_ygx = self.fm.Seps(X_train[k,:], meas, self.geom)
            
            eps_samp = np.random.multivariate_normal(mu_ygx, gamma_ygx)
            Y_train[k,:] = meas + eps_samp

            if (k+1) % 100 == 0:
                print('Sampling: Iteration ', k+1)
        np.save(self.sampleDir + 'Y_train.npy', Y_train)

        for k in range(numTest):
            meas = self.fm.calc_meas(X_test[k,:], self.geom)
            gamma_ygx = self.fm.Seps(X_test[k,:], meas, self.geom)
            
            eps_samp = np.random.multivariate_normal(mu_ygx, gamma_ygx)
            Y_test[k,:] = meas + eps_samp

            if (k+1) % 100 == 0:
                print('Sampling: Iteration ', k+1)
        np.save(self.sampleDir + 'Y_test.npy', Y_test)

    def genReflSampAllPrior(self, surf_mu, surf_gamma, NperPrior=5000):
        # not finished
        numPriors = 8
        Nsamp = NperPrior * numPriors
        ny = surf_mu.shape[1]
        nx = ny + len(atm_mu)

        mu_ygx = np.zeros(ny)
        refl_samp = np.zeros([Nsamp, ny])

        indPr = np.array([0,1,2,3,4,5,6,7,8])

        import matplotlib.pyplot as plt

        for i in range(numPriors):
            for j in range(NperPrior):
                k = i * NperPrior + j

                refl = self.getReflectance(f, i)
                # refl = self.getReflectance(f, np.random.randint(0,4))
                

                mu_x = np.concatenate((surf_mu_scaled[indPr[i]], atm_mu))
                gamma_x = np.zeros([nx, nx])
                gamma_x[:ny, :ny] = surf_gamma_scaled[[indPr[i]],:,:]
                gamma_x[ny:, ny:] = np.diag(atm_gamma)
                cholGammaX = np.linalg.cholesky(gamma_x)
                while x_samp[k,nx-2] < atm_bounds[0,0] or x_samp[k,nx-2] > atm_bounds[0,1] or x_samp[k,nx-1] < atm_bounds[1,0] or x_samp[k,nx-1] > atm_bounds[1,1]:
                    
                    z = np.random.normal(0,1,size=nx)
                    x_samp[k,:] = abs(mu_x + cholGammaX @ z)

                if (k+1) % 100 == 0:
                    print('Sampling: Iteration ', k+1)

       
        np.save(self.sampleDir + 'reflsamp_allprior.npy', x_samp)
            
    def genGMM(self, Nsamp):
        from sklearn import mixture
        from sklearn.model_selection import train_test_split

        # define folder containing data
        folder = '../results/Regression/samples/samplesForGMM/'
        # load wavelengths considered (425 here after sub-selecting wavelengths)
        channel, wl, fwhm = np.loadtxt(folder + 'kelvin_wavelengths.txt').T
        wl = wl * 1000 # convert microns to nanometers

        wlNew = self.setup.wavelengths

        nbands = len(wl)
        print('Nbands:', nbands)
        # define files
        MatTypes  = ['surface_model_ucsb','surface_ArtificialMaterials','surface_Liquids','surface_SoilsAndMixtures','surface_Vegetation']
        nMatType  = len(MatTypes)
        # define array to store data
        dataRaw      = np.zeros((0,nbands))
        for iMatType in range(nMatType):
            # load HDR file for cell type
            hdr_file    = folder + MatTypes[iMatType] + '.hdr'
            file        = open(hdr_file, 'r')
            target_line = [line for line in file if 'lines' in line]
            target_line = target_line[0].split()
            nspec       = int(target_line[2])
            spectra     = np.fromfile(folder + MatTypes[iMatType],
                                dtype=np.float32, count=nspec*nbands)
            spectra     = spectra.reshape((nspec,nbands))
            dataRaw        = np.vstack((dataRaw,spectra))

        print('Total number of spectra/samples: %d' % (dataRaw.shape[0]))
        data = np.zeros((dataRaw.shape[0], len(wlNew)))
        print(data.shape)
        for i in range(dataRaw.shape[0]):
            data[i,:] = np.interp(wlNew, wl, dataRaw[i,:])
        ncomps = 2
        #create mixture model with ncomps components
        gmm = mixture.GaussianMixture(n_components=2)
        gmm.fit(data)
        # print log-likelihood on training data
        print('Log-likelihood on training data:')
        print(gmm.score(data))
        # print weights of components
        print('Weights of components:')
        print(gmm.weights_)
        # generate samples
        samps = gmm.sample(Nsamp)[0]

        # X_train = gmm.sample(24000)[0]
        # X_test = gmm.sample(6000)[0]
        # print(X_train.shape)
        print('Size of samples:')
        # print(samps[0].shape)
        X_train, X_test = train_test_split(samps, test_size=0.25, random_state=42)
        # X_train = X_train[0]
        # X_test = X_test[0]
        print(X_train.shape)
        print(X_test.shape)
        np.save(self.sampleDir + 'X_train_surf.npy', X_train)
        np.save(self.sampleDir + 'X_test_surf.npy', X_test)
        # import matplotlib.pyplot as plt
        # for i in range(10):
        #     print(X_train[i,:])
        #     plt.plot(X_train[i,:])
        # plt.show()

    def addAtm(self):
        X_train_surf = np.load(self.sampleDir + 'X_train_surf.npy')
        X_test_surf = np.load(self.sampleDir + 'X_test_surf.npy')
        Ntrain = X_train_surf.shape[0]
        Ntest = X_test_surf.shape[0]

        atm1_train = np.random.normal(self.setup.mu_x[-2], np.sqrt(self.setup.gamma_x[-2,-2]), (Ntrain,1))
        atm1_test = np.random.normal(self.setup.mu_x[-2], np.sqrt(self.setup.gamma_x[-2,-2]), (Ntest,1))
        atm2_train = np.random.normal(self.setup.mu_x[-1], np.sqrt(self.setup.gamma_x[-1,-1]), (Ntrain,1))
        atm2_test = np.random.normal(self.setup.mu_x[-1], np.sqrt(self.setup.gamma_x[-1,-1]), (Ntest,1))

        print(X_train_surf.shape)
        print(atm1_train.shape)

        X_train = np.concatenate((X_train_surf, atm1_train, atm2_train), axis=1)
        X_test = np.concatenate((X_test_surf, atm1_test, atm2_test), axis=1)

        print(X_train.shape)

        np.save(self.sampleDir + 'X_train.npy', X_train)
        np.save(self.sampleDir + 'X_test.npy', X_test)
        

