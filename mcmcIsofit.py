import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from mcmcLIS import MCMCLIS

class MCMCIsofit:
    '''
    Wrapper class to perform MCMC on Isofit
    Contains functions to perform MCMC sampling
    Integrating LIS capabilities
    '''

    def __init__(self, setup, analysis, Nsamp, burn, x0):

        self.mcmcDir = setup.mcmcDir
        
        # initialize problem parameters
        self.wavelengths = setup.wavelengths
        self.reflectance = setup.reflectance # true reflectance
        self.truth = setup.truth # true state (ref + atm)
        self.radiance = setup.radiance # true radiance
        self.yobs = setup.radNoisy
        self.bands = setup.bands # reflectance indices excluding deep water spectra
        self.bandsX = setup.bandsX # same indices including atm parameters

        # isofit parameters and functions
        self.mu_x = setup.mu_x
        self.gamma_x = setup.gamma_x
        self.mupos_isofit = setup.isofitMuPos
        self.gammapos_isofit = setup.isofitGammaPos
        self.noisecov = setup.noisecov
        
        # linear model
        self.gamma_ygx = analysis.gamma_ygx # error covariance from linear model
        self.linop = analysis.phi # linear operator
        self.linposmean, self.linpos = analysis.posterior(self.yobs) # linear posterior covariance 

        # MCMC parameters to initialize
        self.Nsamp = Nsamp
        self.burn = burn
        self.x0 = x0
        
        self.nx = self.gamma_x.shape[0] # parameter dimension
        self.ny = self.noisecov.shape[0] # data dimension

    def initMCMC(self, LIS=False, rank=427):
        mcmcConfig = {
            "x0": self.x0[:10],
            "Nsamp": self.Nsamp,
            "burn": self.burn,
            "sd": 2.4 ** 2 / rank,
            "LIS": LIS,
            "rank": rank,
            "mu_x": self.mu_x[:10],
            "gamma_x": self.gamma_x[:10,:][:,:10],
            "noisecov": self.noisecov,
            "yobs": self.yobs,
            "linop": self.linop,
            "mcmcDir": self.mcmcDir
            }
        self.mcmc = MCMCLIS(mcmcConfig)

    def runAM(self):
        self.mcmc.adaptm()   

    def calcMeanCov(self):
        return self.mcmc.calcMeanCov()

    def autocorr(self, ind):
        return self.mcmc.autocorr(ind)

    def drawEllipse(self, mean, cov, ax, colour):
        ''' Helper function for twoDimVisual '''

        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='None', edgecolor=colour)
        scale_x = np.sqrt(cov[0, 0]) * 1
        scale_y = np.sqrt(cov[1, 1]) * 1 
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean[0], mean[1])
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse) 
        
    def twoDimVisual(self, MCMCmean, MCMCcov, indX, indY):
        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy')

        fig, ax = plt.subplots()
        '''
        ax.plot(self.truth[indX], self.truth[indY], 'go', label='True reflectance', markersize=10)
        '''
        ax.scatter(x_vals[indX,:], x_vals[indY,:], c='c', s=0.5)

        # plot prior mean/cov
        meanPrior = np.array([self.mu_x[indX], self.mu_x[indY]])
        covPrior = self.gamma_x[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanPrior[0], meanPrior[1], 'kx', label='Prior', markersize=12)
        self.drawEllipse(meanPrior, covPrior, ax, colour='black')
        '''
        # plot Isofit mean/cov
        meanIsofit = np.array([self.mupos_isofit[indX], self.mupos_isofit[indY]])
        covIsofit = self.gammapos_isofit[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanIsofit[0], meanIsofit[1], 'rx', label='Isofit posterior', markersize=12)
        self.drawEllipse(meanIsofit, covIsofit, ax, colour='red')
        '''
        # plot MCMC mean/cov
        meanMCMC = np.array([MCMCmean[indX], MCMCmean[indY]])
        covMCMC = MCMCcov[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanMCMC[0], meanMCMC[1], 'bx', label='MCMC posterior', markersize=12)
        self.drawEllipse(meanMCMC, covMCMC, ax, colour='blue')

        ax.set_title('MCMC - Two Component Visual')
        ax.set_xlabel('Index ' + str(indX))
        ax.set_ylabel('Index ' + str(indY))
        ax.legend()

        return fig

    def diagnostics(self, MCMCmean, MCMCcov, indSet=[10,20,50,100,150,160,250,260,425,426]):
        # assume there are 10 elements in indSet
        # default: indSet = [10,20,50,100,150,160,250,260,425,426]
        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy')
        logpos = np.load(self.mcmcDir + 'logpos.npy')
        numPairs = int(len(indSet) / 2)

        # plot 2D visualization
        for i in range(numPairs):
            fig = self.twoDimVisual(MCMCmean, MCMCcov, indX=indSet[2*i], indY=indSet[2*i+1])
            fig.savefig(self.mcmcDir + '2D_' + str(indSet[2*i]) + '-' + str(indSet[2*i+1]) + '.png')

        # subplot setup
        fig1, axs1 = plt.subplots(5, 2)
        fig2, axs2 = plt.subplots(5, 2)
        xPlot = np.zeros(numPairs * 2, dtype=int)
        yPlot = np.zeros(numPairs * 2, dtype=int)
        xPlot[::2] = range(numPairs)
        xPlot[1::2] = range(numPairs)
        yPlot[1::2] = 1

        for i in range(len(indSet)):
            print('Diagnostics:',indSet[i])
            x_elem = x_vals[i,:]
            xp = xPlot[i]
            yp = yPlot[i]

            # plot trace
            axs1[xp,yp].plot(range(self.Nsamp), x_elem)
            axs1[xp,yp].set_title('Trace - Index ' + str(indSet[i]))

            # plot autocorrelation
            ac = self.autocorr(indSet[i])
            axs2[xp,yp].plot(range(1,len(ac)+1), ac)
            axs2[xp,yp].set_title('Autocorrelation - Index ' + str(indSet[i]))

        fig1.set_size_inches(5, 7)
        fig1.tight_layout()
        fig1.savefig(self.mcmcDir + 'trace.png')
        fig2.set_size_inches(5, 7)
        fig2.tight_layout()
        fig2.savefig(self.mcmcDir + 'autocorr.png')

        # plot logpos
        plt.figure()
        plt.plot(logpos)
        plt.xlabel('Number of Samples')
        plt.ylabel('Log Posterior')
        plt.savefig(self.mcmcDir + 'logpos.png')
        
