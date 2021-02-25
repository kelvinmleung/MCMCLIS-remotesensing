import sys, os, json
import numpy as np
import scipy as s
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class PlotFromFile:

    def __init__(self, mcmcfolder):



        self.paramDir = '../results/Parameters/'
        self.regDir = '../results/Regression/linearmodel/'
        self.mcmcDir = '../results/MCMC/' + mcmcfolder + '/'
        self.mcmcDirNoLIS = '../results/MCMC/N1/'

        self.burn = 5000000
        self.NsampAC = 10000

        self.loadFromFile()
        self.loadMCMC()

        

        ######## CONFIGURE THE TITLES AND STANDARDIZE FONT/SIZE

    def loadFromFile(self):
        self.wavelengths = np.load(self.paramDir + 'wavelengths.npy')
        self.truth = np.load(self.paramDir + 'truth.npy')
        self.radiance = np.load(self.paramDir + 'radiance.npy')

        self.mu_x = np.load(self.paramDir + 'mu_x.npy')
        self.gamma_x = np.load(self.paramDir + 'gamma_x.npy')
        self.isofitMuPos = np.load(self.paramDir + 'isofitMuPos.npy')
        self.isofitGammaPos = np.load(self.paramDir + 'isofitGammaPos.npy')
        self.linMuPos = np.load(self.paramDir + 'linMuPos.npy')
        self.linGammaPos = np.load(self.paramDir + 'linGammaPos.npy')

        self.phi = np.load(self.regDir + 'phi.npy')
        self.meanX = np.load(self.paramDir + 'meanX.npy')
        self.meanY = np.load(self.paramDir + 'meanY.npy')
        self.varX = np.load(self.paramDir + 'varX.npy')
        self.varY = np.load(self.paramDir + 'varY.npy')
    
    def loadMCMC(self):
        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy', mmap_mode='r')
        self.x_vals_plot = x_vals[:,::100]
        self.x_vals_ac = x_vals[:,:self.NsampAC]
        self.MCMCmean = np.mean(x_vals[:,self.burn:], axis=1)
        self.MCMCcov = np.cov(x_vals[:,self.burn:])

        x_vals_noLIS = np.load(self.mcmcDirNoLIS + 'MCMC_x.npy', mmap_mode='r')
        self.x_vals_ac_noLIS = x_vals_noLIS[:,:self.NsampAC]

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

    def plotRegression(self):
        ylinear = self.phi.dot((self.truth - self.meanX) / np.sqrt(self.varX)) * np.sqrt(self.varY) + self.meanY

        plt.figure()
        plt.plot(self.radiance, 'r', label='RT Model')
        plt.plot(ylinear, 'b', label='Linear Model')
        plt.xlabel('Wavelength')
        plt.ylabel('Radiance')
        plt.title('Forward Model Prediction')
        plt.legend()
    
    def plot2Dmarginal(self, indset=[50,90,150,260,410]):
        
        n = len(indset)
        fig, ax = plt.subplots(n, n)
        for i in range(n):
            for j in range(n):
                indX = indset[i]
                indY = indset[j]

                ax[i,j] = self.twoDimVisual(indX, indY, ax[i,j])
                # ax[i,j].set_title('CHANGE TITLE')
                ax[i,j].set_xlabel('Wavelength Channel ' + str(self.wavelengths[indX]))
                ax[i,j].set_ylabel('Wavelength Channel ' + str(self.wavelengths[indY]))
        # ax.set_title('2D Marginal Plots ')
        fig.savefig(self.mcmcDir + '2Dmarginal.png', dpi=300)

    # def plot2ac()
    # def plotkdcontour()
    # def autocorr()

    def plotposmean(self):
        plt.figure()
        # self.plotbands(self.mu_x[:425], 'r', label='Prior')
        self.plotbands(self.truth[:425], 'b.',label='True Reflectance')
        self.plotbands(self.isofitMuPos[:425],'k.', label='MAP Estimate')
        # self.plotbands(self.linMuPos[:425], 'm.',label='Linear Posterior')
        self.plotbands(self.MCMCmean[:425], 'c.',label='MCMC Posterior')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.title('Posterior Mean - Surface Reflectance')
        plt.grid()
        plt.legend()
        # plt.savefig(self.mcmcDir + 'reflMean.png', dpi=300)

        plt.figure()
        plt.plot(self.truth[425], self.truth[426], 'bo',label='True Atm Param')
        # plt.plot(self.mu_x[425], self.mu_x[426], 'r.',label='Prior')
        plt.plot(self.isofitMuPos[425],self.isofitMuPos[426],'k.', label='MAP Estimate')
        # plt.plot(self.linMuPos[425], self.linMuPos[426],'mx',label='Linear Posterior')
        plt.plot(self.MCMCmean[425], self.MCMCmean[426], 'cx',label='MCMC Posterior')
        plt.xlabel('AOD550')
        plt.ylabel('H2OSTR')
        plt.title('Posterior Mean - Atmospheric Parameters')
        plt.grid()
        plt.legend()

    def plotposvar(self):
        priorVar = np.diag(self.gamma_x)
        isofitVar = np.diag(self.isofitGammaPos)
        # linearVar = np.diag(self.linGammaPos)
        MCMCVar = np.diag(self.MCMCcov)
        plt.figure()
        self.plotbands(priorVar[:425], 'b.',label='Prior', axis='semilogy')
        self.plotbands(isofitVar[:425],'k.', label='MAP Estimate', axis='semilogy')
        self.plotbands(MCMCVar[:425], 'c.',label='MCMC Posterior', axis='semilogy')
        plt.xlabel('Wavelength')
        plt.ylabel('Marginal Variance')
        plt.title('Posterior Variance - Surface Reflectance')
        plt.grid()
        plt.legend()
        # plt.savefig(self.mcmcDir + 'reflVar.png', dpi=300)

        labels = ['425 - AOD550', '426 - H2OSTR']
        x = np.arange(len(labels))  # the label locations
        width = 0.175
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, priorVar[425:], width, label='Prior')
        rects2 = ax.bar(x, isofitVar[425:], width, label='MAP Estimate')
        rects3 = ax.bar(x + width, MCMCVar[425:], width, label='MCMC Posterior')
        ax.set_yscale('log')
        ax.set_ylabel('Marginal Variance')
        ax.set_title('Posterior Variance - Atmospheric Parameters')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        # fig.savefig(self.mcmcDir + 'atmVar.png', dpi=300)


    

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

    def twoDimVisual(self, indX, indY, ax):
        x_vals = self.x_vals_plot

        ax.plot(self.truth[indX], self.truth[indY], 'go', label='True reflectance', markersize=10)     
        ax.scatter(x_vals[indX,:], x_vals[indY,:], c='c', s=0.5)

        # plot Isofit mean/cov
        meanIsofit = np.array([self.isofitMuPos[indX], self.isofitMuPos[indY]])
        covIsofit = self.isofitGammaPos[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanIsofit[0], meanIsofit[1], 'rx', label='Isofit posterior', markersize=12)
        self.drawEllipse(meanIsofit, covIsofit, ax, colour='red')
        
        # plot MCMC mean/cov
        meanMCMC = np.array([self.MCMCmean[indX], self.MCMCmean[indY]])
        covMCMC = self.MCMCcov[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanMCMC[0], meanMCMC[1], 'bx', label='MCMC posterior', markersize=12)
        self.drawEllipse(meanMCMC, covMCMC, ax, colour='blue')
        
        # ax.set_title('MCMC - Two Component Visual')
        # ax.set_xlabel('Index ' + str(indX))
        # ax.set_ylabel('Index ' + str(indY))
        ax.legend()
        return ax
    

    










