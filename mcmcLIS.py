import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class MCMCLIS:
    '''
    Contains functions to perform MCMC sampling
    Integrating LIS capabilities
    '''

    def __init__(self, config):

        self.unpackConfig(config)
        self.nx = self.gamma_x.shape[0] # parameter dimension
        self.ny = self.noisecov.shape[0] # data dimension

        # initialize chain and proposal covariance
        self.x0 = self.x0 - self.mu_x

        self.invGammaX = np.linalg.inv(self.gamma_x)
        self.invNoiseCov = np.linalg.inv(self.noisecov)

        if self.LIS == True:
            # compute projection matrices
            self.phi, self.theta, self.proj, self.phiComp, self.thetaComp, self.projComp = self.LISproject() 

             # project x0 and proposal covariance to LIS subspace
            self.x0 = self.theta.T @ self.x0
            self.propcov = self.theta.T @ self.propcov @ self.theta


    def unpackConfig(self, config):
        self.x0 = config["x0"]              # initial value of chain
        self.Nsamp = config["Nsamp"]        # total number of MCMC samples
        self.burn = config["burn"]          # number of samples discarded as burn-in
        self.sd = config["sd"]              # proposal covariance parameter
        self.propcov = config["propcov"]    # initial proposal covariance
        self.LIS = config["LIS"]            # LIS or no LIS
        self.rank = config["rank"]          # rank of problem
        self.mu_x = config["mu_x"]          # prior mean
        self.gamma_x = config["gamma_x"]    # prior covariance
        self.noisecov = config["noisecov"]  # data noise covariance
        self.fm = config["fm"]              # forward model
        self.geom = config["geom"]          # geometry model
        self.linop = config["linop"]        # linear operator
        self.yobs = config["yobs"]          # radiance observation
        self.mcmcDir = config["mcmcDir"]    # directory to save data

            
    def LISproject(self):
        ### Compute LIS projection matrices ###

        # compute Hessian
        cholPr = np.linalg.cholesky(self.gamma_x) # cholesky decomp of prior covariance
        H = self.linop.T @ self.invNoiseCov @ self.linop # Hessian
        Hn = cholPr.T @ H @ cholPr 

        print('Solving generalized eigenvalue problem...')
        
        # solve eigenvalue problem, sort
        eigvec, eigval, p = np.linalg.svd(Hn)
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        V = eigvec[:,idx]

        # plt.semilogy(eigval)
        # plt.title('LIS Eigenvalue Decay')
        # plt.grid()
        # plt.show()

        # LIS subspace
        VLIS = V[:,:self.rank] 
        phi = cholPr @ VLIS
        theta = np.linalg.inv(cholPr.T) @ VLIS
        proj = phi @ theta.T

        # complementary subspace
        VComp = V[:,self.rank:]
        phiComp = cholPr @ VComp
        thetaComp = np.linalg.inv(cholPr.T) @ VComp
        projComp = phiComp @ thetaComp.T

        return phi, theta, proj, phiComp, thetaComp, projComp


    def logpos(self, x):
        ''' Calculate log posterior '''
        # input x is zero-mean in LIS parameter space
        if self.LIS == True:
            # gammax = np.identity(np.size(x)) # prior of normalized LIS space
            # logprior = -1/2 * x.dot(np.linalg.solve(gammax, x))
            logprior = -1/2 * x.dot(x)
            x = self.phi @ x  + self.mu_x # project back to original (physical) space
        else:
            # logprior = -1/2 * x.dot(np.linalg.solve(self.gamma_x, x))
            logprior = -1/2 * (x @ self.invGammaX @ x.T)
            x = x + self.mu_x

        meas = self.fm.calc_rdn(x, self.geom) # apply forward model
        tLH = self.yobs - meas
        loglikelihood = -1/2 * (tLH @ self.invNoiseCov @ tLH.T)


        # for fixed atm
        # xFull = np.concatenate((x, [0.05,1.75]))
        # meas = self.fm.calc_rdn(xFull, self.geom) # apply forward model
        # tLH = self.yobs - meas
        # loglikelihood = -1/2 * tLH.dot(np.linalg.solve(self.noisecov, tLH))
        

        # plt.figure(100)
        # plt.plot(meas, 'r')
        # plt.plot(self.yobs, 'b')
        # plt.ylim([-0.1, 15])
        # plt.show(block=False)
        # plt.pause(0.0001)
        # plt.close()
        
        # if x[425] < 0 or x[426] < 0:
        #     print('ATM parameter is negative')
        #     loglikelihood = -np.Inf
        #     print(x[425:])
        
        return logprior #+ loglikelihood 

    def proposal(self, mean, covCholesky):
        ''' Sample proposal from a normal distribution '''
        zx = np.random.normal(0, 1, size=mean.size)
        z = mean + covCholesky @ zx
        return z

    def alpha(self, x, z):
        ''' Calculate acceptance ratio '''
        logposZ = self.logpos(z)
        logposX = self.logpos(x)
        ratio = logposZ - logposX
        # return both acceptance ratio and logpos
        return np.minimum(1, np.exp(ratio)), logposZ, logposX

    def adaptm(self):
        ''' Run Adaptive-Metropolic MCMC algorithm '''
        x_vals = np.zeros([self.x0.size, self.Nsamp]) # store all samples
        logpos = np.zeros(self.Nsamp) # store the log posterior values
        accept = np.zeros(self.Nsamp, dtype=int)

        x = self.x0
        propChol = np.linalg.cholesky(self.propcov) # cholesky decomp of proposal cov
        eps = 1e-10

        for i in range(self.Nsamp):
            z = self.proposal(x, propChol)
            
            alpha, logposZ, logposX = self.alpha(x, z)
            if np.random.random() < alpha:
                x = z 
                logposX = logposZ
                accept[i] = 1
            x_vals[:,i] = x
            logpos[i] = logposX
            
            # print progress
            if (i+1) % 500 == 0: 
                print('Sample: ', i+1)
                print('   Accept Rate: ', np.mean(accept[i-499:i]))
                propChol = np.linalg.cholesky(self.propcov) # update chol of propcov
                print(np.linalg.norm(propChol))

                # plot the proposal
                # self.plotProposal(z)
                
            # change proposal covariance
            if i == 999:
                self.propcov = self.sd * (np.cov(x_vals[:,:1000]) + eps * np.identity(len(x)))
                meanXprev = np.mean(x_vals[:,:1000],1)
            elif i >= 1000:
                meanX = i / (i + 1) * meanXprev + 1 / (i + 1) * x_vals[:,i]
                self.propcov = (i-1) / i * self.propcov + self.sd / i * (i * np.outer(meanXprev, meanXprev) - (i+1) * np.outer(meanX, meanX) + np.outer(x_vals[:,i], x_vals[:,i]) + eps * np.identity(len(x)))
                meanXprev = meanX
                # propChol = np.linalg.cholesky(self.propcov) # update chol of propcov

            # if i == 999:
            #     self.propcov = self.sd * (np.cov(x_vals[:,500:1000]) + eps * np.identity(len(x)))
            #     meanXprev = np.mean(x_vals[:,500:1000],1)
            # elif i >= 1000:
            #     j = i - 500
            #     meanX = j / (j + 1) * meanXprev + 1 / (j + 1) * x_vals[:,i]
            #     self.propcov = (j-1) / j * self.propcov + self.sd / j * (j * np.outer(meanXprev, meanXprev) - (j+1) * np.outer(meanX, meanX) + np.outer(x_vals[:,i], x_vals[:,i]) + eps * np.identity(len(x)))
            #     meanXprev = meanX
        
        # post processing, store MCMC chain
        x_vals_full = np.zeros([self.nx, self.Nsamp])
        if self.LIS == True:
            # add samples of xComp to chain, project back to full subspace
            nComp = np.shape(self.phiComp)[1] # size of complementary subspace
            for i in range(self.Nsamp):
                xComp = self.proposal(np.zeros(nComp), np.identity(nComp))
                x_vals_full[:,i] = self.phi @ x_vals[:,i] + self.phiComp @ xComp + self.mu_x
        else:
            for i in range(self.Nsamp):
                x_vals_full[:,i] = x_vals[:,i] + self.mu_x

        np.save(self.mcmcDir + 'MCMC_x.npy', x_vals_full)
        np.save(self.mcmcDir + 'logpos.npy', logpos)
        np.save(self.mcmcDir + 'acceptance.npy', accept)
        return x_vals  
    
    def plotProposal(self, z):
        if self.LIS == True:
            plt.plot(self.phi @ z + self.mu_x)
        else:
            plt.plot(z + self.mu_x)
        plt.ylim([-0.1, 0.8])
        plt.show(block=False)
        plt.pause(0.001)
        plt.close()

    def calcMeanCov(self):
        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy')
        x_ref = x_vals[:, self.burn:]
        nx = x_ref.shape[0]
        mean = np.mean(x_ref, axis=1)
        cov = np.cov(x_ref)
        return mean, cov

    def autocorr(self, ind):
        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy')
        x_elem = x_vals[ind,:]
        Nsamp = min(self.Nsamp, 20000)
        meanX = np.mean(x_elem)
        varX = np.var(x_elem)
        ac = np.zeros(Nsamp-1)

        for k in range(Nsamp-1):
            extra = min(4000, self.Nsamp - Nsamp)
            cov = np.cov(x_elem[:Nsamp-k + extra], x_elem[k:Nsamp+extra])
            ac[k] = cov[1,0] / varX
        return ac


        
