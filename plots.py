
import sys, os, json
import numpy as np
import scipy as s
from scipy.io import loadmat
import matplotlib.pyplot as plt

class Plots:


    def __init__(self, setup, r, a, m, mcmcNoLIS=''):
        self.setup = setup
        self.m = m

        self.wavelengths = setup.wavelengths

        self.sampleDir = setup.sampleDir
        self.regDir = setup.regDir
        self.analysisDir = setup.analysisDir
        self.mcmcDir = setup.mcmcDir
        self.mcmcDir2 = '../results/MCMC/' + mcmcNoLIS + '/'

        # reg things to plot


        # eigval/vec to plot
        # self.eigval
        # self.eigvec

        # mean and covariance
        self.truth = setup.truth
        self.mu_x = setup.mu_x
        self.gamma_x = setup.gamma_x
        self.isofitMuPos = setup.isofitMuPos
        self.isofitGammaPos = setup.isofitGammaPos
        self.linMuPos = m.linMuPos
        self.linGammaPos = m.linGammaPos

        self.Nsamp = m.Nsamp
        self.MCMCmean, self.MCMCcov = m.calcMeanCov()

        self.burn = 4000000
        self.autocorrMax = 10000

        self.indset = [20,30, 150,160, 250,260]

    def readFile(self, mcmcdir):
        x_vals_all = np.load(mcmcdir + 'MCMC_x.npy')
        x_vals = x_vals_all[:, self.burn:]
        x_vals_ac = x_vals_all[:, self.autocorrMax:]
        return x_vals, x_vals_ac

    def plotPosterior(self):
        self.posmeansurf()
        self.posvarsurf()
        self.poserrorsurf()
        self.posmeanatm()
        self.posvarsurf()
        self.poserrorsurf()

    def CSE21plots_double(self):
        x_vals1, x_vals_ac1 = self.readFile(self.mcmcDir)
        x_vals2, x_vals_ac2 = self.readFile(self.mcmcDir2)
        
        self.plot2Dmarginal(x_vals1)

        

    def twoDimVisual(self, indX, indY, x_vals):
        fig, ax = plt.subplots()
        ax.plot(self.truth[indX], self.truth[indY], 'go', label='True reflectance', markersize=10)     
        ax.scatter(x_vals[indX,:], x_vals[indY,:], c='c', s=0.5)

        # plot Isofit mean/cov
        meanIsofit = np.array([self.mupos_isofit[indX], self.mupos_isofit[indY]])
        covIsofit = self.gammapos_isofit[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanIsofit[0], meanIsofit[1], 'rx', label='Isofit posterior', markersize=12)
        self.drawEllipse(meanIsofit, covIsofit, ax, colour='red')
        
        # plot MCMC mean/cov
        meanMCMC = np.array([self.MCMCmean[indX], self.MCMCmean[indY]])
        covMCMC = self.MCMCcov[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanMCMC[0], meanMCMC[1], 'bx', label='MCMC posterior', markersize=12)
        self.drawEllipse(meanMCMC, covMCMC, ax, colour='blue')
        
        ax.set_title('MCMC - Two Component Visual')
        ax.set_xlabel('Index ' + str(indX))
        ax.set_ylabel('Index ' + str(indY))
        ax.legend()

        return fig, ax
    
    def plot2Dmarginal(self, x_vals):
        
        numPairs = int(len(self.indset) / 2)

        # plot 2D visualization - surf
        for i in range(numPairs):
            indX = self.indset[2*i]
            indY = self.indset[2*i+1]
            fig, ax = self.twoDimVisual(self.MCMCmean, self.MCMCcov, indX=indX, indY=indY)
            ax.set_title('CHANGE TITLE')
            ax.set_xlabel('Wavelength Channel ' + str(self.wavelengths[indX]))
            ax.set_ylabel('Wavelength Channel ' + str(self.wavelengths[indY]))
            fig.savefig(self.mcmcDir + '2D_' + str(indX) + '-' + str(indY) + '.png', dpi=300)

        # plot 2D atm
        indX = self.indset[2*i]
        indY = self.indset[2*i+1]
        fig, ax = self.m.twoDimVisual(self.MCMCmean, self.MCMCcov, indX=425, indY=426)
        ax.set_title('CHANGE TITLE')
        ax.set_xlabel('AOD550')
        ax.set_ylabel('H2OSTR')
        fig.savefig(self.mcmcDir + '2D_atm.png', dpi=300)

    def autocorr(self, x_vals_ac):
        
        x_vals = x_vals_ac 
        x_elem = x_vals[ind,:]
        Nsamp = self.autocorrMax
        meanX = np.mean(x_elem)
        varX = np.var(x_elem)
        ac = np.zeros(Nsamp-1)

        for k in range(Nsamp-1):
            cov = np.cov(x_elem[:Nsamp-k], x_elem[k:Nsamp])
            ac[k] = cov[1,0] / varX

        return ac[:int(Nsamp/2)]

    def plot2ac(self, ac, ac2):

        for i in range(len(self.indset)):
            print('Diagnostics:',self.indset[i])
            x_elem = x_vals[indSet[i],:]
            xp = xPlot[i]
            yp = yPlot[i]

            # plot autocorrelation
            axs2[xp,yp].plot(range(1,len(ac)+1), ac1)
            axs2[xp,yp].plot(range(1,len(ac2)+1), ac2)
            axs2[xp,yp].set_title('Autocorrelation - Index ' + str(indSet[i]))

        fig2.set_size_inches(5, 7)
        fig2.tight_layout()
        fig2.savefig(self.mcmcDir + 'autocorr.png', dpi=300)
        

    
    def plotbands(self, y, linestyle, linewidth=1, label='', axis='normal'):
        return self.setup.plotbands(y, linestyle, linewidth, label, axis)    

    def regRadiance(self):
        return

    def regError(self):
        return

    def regSparsity(self):
        return

    def eigenplot(self):
        return
    

    def posmeansurf(self):
        plt.figure()
        # self.plotbands(self.mu_x[:425], 'r', label='Prior')
        self.plotbands(self.truth[:425], 'b.',label='True Reflectance')
        self.plotbands(self.isofitMuPos[:425],'k.', label='Isofit Posterior')
        self.plotbands(self.linMuPos[:425], 'm.',label='Linear Posterior')
        self.plotbands(self.MCMCmean[:425], 'c.',label='MCMC Posterior')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.title('Posterior Mean Comparison')
        plt.grid()
        plt.legend()
        plt.savefig(self.mcmcDir + 'reflMean.png', dpi=300)
    
    def poserrorsurf(self):
        plt.figure()
        isofitError = abs(self.isofitMuPos[:425] - self.truth[:425]) / abs(self.truth[:425])
        linError = abs(self.linMuPos[:425] - self.truth[:425]) / abs(self.truth[:425])
        mcmcError = abs(self.MCMCmean[:425] - self.truth[:425]) / abs(self.truth[:425])
        self.plotbands(isofitError,'k.', label='Isofit Posterior',axis='semilogy')
        self.plotbands(linError, 'm.',label='Linear Posterior',axis='semilogy')
        self.plotbands(mcmcError, 'c.',label='MCMC Posterior',axis='semilogy')
        plt.xlabel('Wavelength')
        plt.ylabel('Relative Error')
        plt.title('Error in Posterior Mean')
        plt.grid()
        plt.legend()
        plt.savefig(self.mcmcDir + 'reflError.png', dpi=300)
    
    def posvarsurf(self):
        priorVar = np.diag(self.gamma_x)
        isofitVar = np.diag(self.isofitGammaPos)
        linearVar = np.diag(gamma_xgyLin)
        MCMCVar = np.diag(MCMCcov)
        plt.figure()
        self.plotbands(priorVar[:425], 'b.',label='Prior', axis='semilogy')
        self.plotbands(isofitVar[:425],'k.', label='Isofit Posterior', axis='semilogy')
        self.plotbands(linearVar[:425], 'm.',label='Linear Posterior', axis='semilogy')
        self.plotbands(MCMCVar[:425], 'c.',label='MCMC Posterior', axis='semilogy')
        plt.xlabel('Wavelength')
        plt.ylabel('Variance')
        plt.title('Marginal Variance Comparison')
        plt.grid()
        plt.legend()
        plt.savefig(self.mcmcDir + 'reflVar.png', dpi=300)
        return

    def posmeanatm(self):
        plt.figure()
        plt.plot(self.truth[425], self.truth[426], 'bo',label='True Reflectance')
        plt.plot(self.mu_x[425], self.mu_x[426], 'r.',label='Prior')
        plt.plot(self.isofitMuPos[425],self.isofitMuPos[426],'k.', label='Isofit Posterior')
        plt.plot(self.linMuPos[425], self.linMuPos[426],'mx',label='Linear Posterior')
        plt.plot(self.MCMCmean[425], self.MCMCmean[426], 'cx',label='MCMC Posterior')
        plt.xlabel('AOT550')
        plt.ylabel('H2OSTR')
        plt.grid()
        plt.legend()
        plt.savefig(self.mcmcDir + 'atmMean.png', dpi=300)
        return

    def posvaratm(self):
        isofitErrorAtm = abs(self.isofitMuPos[425:] - self.truth[425:]) / abs(self.truth[425:])
        linErrorAtm = abs(self.linMuPos[425:] - self.truth[425:]) / abs(self.truth[425:])
        mcmcErrorAtm = abs(self.MCMCmean[425:] - self.truth[425:]) / abs(self.truth[425:])
        labels = ['425 - AOD550', '426 - H2OSTR']
        x = np.arange(len(labels))  # the label locations
        width = 0.175
        fig, ax = plt.subplots()
        rects2 = ax.bar(x - width, isofitErrorAtm, width, label='Isofit Posterior')
        rects3 = ax.bar(x, linErrorAtm, width, label='Linear Posterior')
        rects4 = ax.bar(x + width, mcmcErrorAtm, width, label='MCMC Posterior')
        ax.set_yscale('log')
        ax.set_ylabel('Relative Error')
        ax.set_title('Error in Atm Parameters')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.savefig(self.mcmcDir + 'atmError.png', dpi=300)
        return

    def poserroratm(self):
        labels = ['425 - AOD550', '426 - H2OSTR']
        x = np.arange(len(labels))  # the label locations
        width = 0.175
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - 3*width/2, priorVar[425:], width, label='Prior')
        rects2 = ax.bar(x - width/2, isofitVar[425:], width, label='Isofit Posterior')
        rects3 = ax.bar(x + width/2, linearVar[425:], width, label='Linear Posterior')
        rects4 = ax.bar(x + 3*width/2, MCMCVar[425:], width, label='MCMC Posterior')
        ax.set_yscale('log')
        ax.set_ylabel('Variance')
        ax.set_title('Marginal Variance of Atm')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.savefig(self.mcmcDir + 'atmVar.png', dpi=300)

    def autocorr(self):
        return

    
    




