import sys, os, json
import numpy as np
import scipy as s
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class PlotFromFile:

    def __init__(self, mcmcfolder):

        self.mcmcfolder = mcmcfolder

        self.paramDir = '../results/Parameters/'
        self.regDir = '../results/Regression/linearmodel/'
        self.mcmcDir = '../results/MCMC/' + mcmcfolder + '/'
        self.mcmcDirNoLIS = '../results/MCMC/N1/'

        self.Nsamp = 6000000
        self.burn = 4000000
        self.NsampAC = 10000
        self.numPlotAC = 2000

        self.loadFromFile()
        self.loadMCMC()

        
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

        # self.x_vals_plot = x_vals[:,::50]
        # self.MCMCmean = np.mean(x_vals[:,self.burn:], axis=1)
        # self.MCMCcov = np.cov(x_vals[:,self.burn:])

        self.x_vals_plot = x_vals[:,self.burn::50]
        self.MCMCmean = np.mean(x_vals[:,self.burn:], axis=1)
        self.MCMCcov = np.cov(x_vals[:,self.burn:])

        self.x_vals_ac = x_vals[:,:self.NsampAC]

        np.save(self.paramDir + 'MCMCmean' + str(self.mcmcfolder) + '.npy', self.MCMCmean)
        np.save(self.paramDir + 'MCMCcov' + str(self.mcmcfolder) + '.npy', self.MCMCcov)

        x_vals_noLIS = np.load(self.mcmcDirNoLIS + 'MCMC_x.npy', mmap_mode='r')
        self.x_vals_ac_noLIS = x_vals_noLIS[:,:self.NsampAC]

    def plotbands(self, y, linestyle, linewidth=2, label='', axis='normal'):
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
        plt.plot(self.wavelengths, self.radiance, 'r', linewidth=1.5, label='RT Model')
        plt.plot(self.wavelengths, ylinear, 'b', linewidth=1.5, label='Linear Model')
        plt.xlabel('Wavelength')
        plt.ylabel('Radiance')
        plt.title('Forward Model Prediction')
        plt.legend()
    
    def plot2Dmarginal(self, indset1=[100], indset2=[30,101,260]): #,250,410
        
        n = len(indset1)
        m = len(indset2)
        fig, ax = plt.subplots(n, m)
        
        for i in range(n):
            for j in range(m):
                indX = indset1[i]
                indY = indset2[j]

                ax[j] = self.twoDimVisual(indY, indX, ax[j])
                # ax[i,j].set_title('CHANGE TITLE')
                # ax[i,j].set_xlabel('Wavelength Channel ' + str(self.wavelengths[indY]))
                # ax[i,j].set_ylabel('Wavelength Channel ' + str(self.wavelengths[indX]))
        # ax.set_title('2D Marginal Plots ')
        # fig.savefig(self.mcmcDir + '2Dmarginal.png', dpi=300)

        ax[0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[0]]) + ' nm')
        # ax[1,0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[1]]) + ' nm')
        # ax[2,0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[2]]) + ' nm')

        ax[0].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[0]]) + ' nm')
        ax[1].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[1]]) + ' nm')
        ax[2].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[2]]) + ' nm')
        # ax[2,0].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[0]]) + ' nm')
        # ax[2,1].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[1]]) + ' nm')
        # ax[2,2].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[2]]) + ' nm')
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')

    def plotkdcontour(self, indX, indY):

        x = self.x_vals_plot[indX,:]
        y = self.x_vals_plot[indY,:]
        xmin, xmax = 0.61, 0.66
        ymin, ymax = 0.615, 0.665

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        f = f / np.max(f) # normalize

        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        levs = [0, 0.02, 0.05, 0.1, 0.25, 0.5, 1]
        levs = [0, 0.05, 0.2, 0.5, 1]
        # Contourf plot
        cfset = ax.contourf(xx, yy, f, levels=levs, cmap='Blues') 
        cset = ax.contour(xx, yy, f, levels=levs, colors='k') ##############################ADD INLINE
        plt.clabel(cset, levs, fontsize='smaller')

        # plot truth, isofit, and mcmc mean
        meanIsofit = np.array([self.isofitMuPos[indX], self.isofitMuPos[indY]])
        meanMCMC = np.array([self.MCMCmean[indX], self.MCMCmean[indY]])
        ax.plot(self.truth[indX], self.truth[indY], 'go', label='Truth', markersize=8)  
        ax.plot(meanIsofit[0], meanIsofit[1], 'rx', label='MAP', markersize=12)
        ax.plot(meanMCMC[0], meanMCMC[1], 'kx', label='MCMC', markersize=12)

        # Label plot
        # ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel(r'$\lambda = $' + str(self.wavelengths[indX]) + ' nm')
        ax.set_ylabel(r'$\lambda = $' + str(self.wavelengths[indY]) + ' nm')
        ax.legend()
        fig.colorbar(cfset)


    def plot2ac(self, indset=[120,250,410]):

        fig, axs = plt.subplots(1, len(indset))

        for i in range(len(indset)):
            # print('Autocorr:', indset[i])

            ac = self.autocorr(self.x_vals_ac[indset[i],:])
            ac2 = self.autocorr(self.x_vals_ac_noLIS[indset[i],:])

            ac = ac[:self.numPlotAC]
            ac2 = ac2[:self.numPlotAC]

            print('Index:', indset[i])
            print('ESS LIS:', self.ESS(ac))
            print('ESS No LIS:', self.ESS(ac2))

            # plot autocorrelation
            axs[i].plot(range(1,len(ac)+1), ac, 'b', label='LIS r = 100')
            axs[i].plot(range(1,len(ac2)+1), ac2, 'r', label='No LIS')
            if indset[i] < 425:
                axs[i].set_title(r'$\lambda = $' + str(self.wavelengths[indset[i]]) + ' nm')
            elif indset[i] == 425:
                axs[i].set_title('AOD')
            elif indset[i] == 426:
                axs[i].set_title('H2O')
        
        axs[0].set_xlabel('Lag', fontsize=14)
        axs[0].set_ylabel('Autocorrelation', fontsize=14)
        
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', fontsize=14)
        # fig2.savefig(self.mcmcDir + 'autocorr.png', dpi=300)

    
    def autocorr(self, x_elem):
        Nsamp = self.NsampAC
        meanX = np.mean(x_elem)
        varX = np.var(x_elem)
        ac = np.zeros(Nsamp-1)

        for k in range(Nsamp-1):
            cov = np.cov(x_elem[:Nsamp-k], x_elem[k:Nsamp])
            ac[k] = cov[1,0] / varX

        return ac

    def ESS(self, ac):
        denom = 0
        for i in range(len(ac)):
            denom = denom + ac[i]
        return self.Nsamp / (1 + 2 * denom)


    def plotposmean(self):
        plt.figure()
        # self.plotbands(self.mu_x[:425], 'r', label='Prior')
        self.plotbands(self.truth[:425], 'r',label='Truth')
        self.plotbands(self.isofitMuPos[:425],'k', label='MAP Estimate')
        # self.plotbands(self.linMuPos[:425], 'm.',label='Linear Posterior')
        self.plotbands(self.MCMCmean[:425], 'b',label='MCMC Posterior')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.title('Posterior Mean - Surface Reflectance')
        plt.legend()
        # plt.savefig(self.mcmcDir + 'reflMean.png', dpi=300)

        plt.figure()
        plt.plot(self.truth[425], self.truth[426], 'rx',label='Truth')
        # plt.plot(self.mu_x[425], self.mu_x[426], 'r.',label='Prior')
        plt.plot(self.isofitMuPos[425],self.isofitMuPos[426],'ko', label='MAP Estimate')
        # plt.plot(self.linMuPos[425], self.linMuPos[426],'mx',label='Linear Posterior')
        plt.plot(self.MCMCmean[425], self.MCMCmean[426], 'bo',label='MCMC Posterior')
        plt.xlabel('AOD550')
        plt.ylabel('H2OSTR')
        plt.title('Posterior Mean - Atmospheric Parameters')
        plt.xlim([0, 0.2])
        plt.ylim([2, 3])
        plt.legend()

    def plotposvar(self):
        priorVar = np.diag(self.gamma_x)
        isofitVar = np.diag(self.isofitGammaPos)
        # linearVar = np.diag(self.linGammaPos)
        MCMCVar = np.diag(self.MCMCcov)
        plt.figure()
        self.plotbands(priorVar[:425], 'r',label='Prior', axis='semilogy')
        self.plotbands(isofitVar[:425],'k', label='Laplace Approx', axis='semilogy')
        self.plotbands(MCMCVar[:425], 'b',label='MCMC Posterior', axis='semilogy')
        plt.xlabel('Wavelength')
        plt.ylabel('Marginal Variance')
        plt.title('Posterior Variance - Surface Reflectance')
        plt.legend()
        # plt.savefig(self.mcmcDir + 'reflVar.png', dpi=300)

        labels = ['AOD550', 'H2OSTR']
        x = np.arange(len(labels))  # the label locations
        width = 0.175
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, priorVar[425:], width, color='red', label='Prior')
        rects2 = ax.bar(x, isofitVar[425:], width, color='black', label='Laplace Approx')
        rects3 = ax.bar(x + width, MCMCVar[425:], width, color='blue', label='MCMC Posterior')
        ax.set_yscale('log')
        ax.set_ylabel('Marginal Variance')
        ax.set_title('Posterior Variance - Atmospheric Parameters')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        # fig.savefig(self.mcmcDir + 'atmVar.png', dpi=300)

    def plotPosSparsity(self, tol):

        deadbands = list(range(185,215)) + list(range(281,315)) + list(range(414,425))
        cov = self.MCMCcov
        for i in deadbands:
            cov[i,:] = np.zeros(cov.shape[0])
            cov[:,i] = np.zeros(cov.shape[0])

        plt.figure()
        plt.spy(cov, color='b', precision=tol, markersize=2)
        plt.title('Sparsity Plot of Posterior Covariance - Tolerance = ' + str(tol))

    def plotPosCovRow(self, indset=[120,250,410]):
        
        for i in indset:
            plt.figure()
            self.plotbands(self.MCMCcov[i,:], 'b.', axis='semilogy')
            plt.title('Posterior Covariance - Row of index ' + str(i) + ', wavelength=' + str(self.wavelengths[i]))
            plt.xlabel('Wavelength')
            plt.ylabel('Value of Covariance Matrix')




    def plotCompareRank(self):
        
        meanB8 = np.load(self.paramDir + 'MCMCmeanB8.npy')
        covB8 = np.load(self.paramDir + 'MCMCcovB8.npy')
        meanC8 = np.load(self.paramDir + 'MCMCmeanC8.npy')
        covC8 = np.load(self.paramDir + 'MCMCcovC8.npy')
        
        varB8 = np.diag(covB8)
        varC8 = np.diag(covC8)
        varMAP = np.diag(self.isofitGammaPos)

        errMeanMAP = abs(self.isofitMuPos - self.truth) / self.truth
        errMeanB8 = abs(meanB8 - self.truth) / self.truth
        errMeanC8 = abs(meanC8 - self.truth) / self.truth
        '''
        # posterior mean error
        plt.figure()
        self.plotbands(errMeanMAP,'k.', label='MAP Estimate', axis='semilogy')
        self.plotbands(errMeanB8, 'r.',label='LIS r=100', axis='semilogy')
        self.plotbands(errMeanC8, 'b.',label='LIS r=175', axis='semilogy')
        plt.xlabel('Wavelength')
        plt.ylabel('Relative Error')
        plt.title('Error in Posterior Mean - Surface Reflectance')
        plt.legend()

        labels = ['425 - AOD550', '426 - H2OSTR']
        x = np.arange(len(labels))  # the label locations
        width = 0.175
        fig, ax = plt.subplots()
        rects2 = ax.bar(x - width, errMeanMAP[425:], width, color='black', label='MAP Estimate')
        rects1 = ax.bar(x, errMeanB8[425:], width, color='red', label='LIS r=100')
        rects3 = ax.bar(x + width, errMeanC8[425:], width, color='blue', label='LIS r=175')
        ax.set_yscale('log')
        ax.set_ylabel('Relative Error')
        ax.set_title('Error in Posterior Mean - Atm.')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        '''
        # posterior variance
        plt.figure()
        self.plotbands(varMAP,'k', label='Laplace Approx', axis='semilogy')
        self.plotbands(varB8, 'r',label='LIS r=100', axis='semilogy')
        self.plotbands(varC8, 'b',label='LIS r=175', axis='semilogy')
        plt.xlabel('Wavelength')
        plt.ylabel('Marginal Variance')
        plt.title('Posterior Variance - Surface Reflectance')
        plt.legend()

        labels = ['425 - AOD550', '426 - H2OSTR']
        x = np.arange(len(labels))  # the label locations
        width = 0.175
        fig, ax = plt.subplots()
        rects2 = ax.bar(x - width, varMAP[425:], width, color='black', label='Laplace Approx')
        rects1 = ax.bar(x, varB8[425:], width, color='red', label='LIS r=100')
        rects3 = ax.bar(x + width, varC8[425:], width, color='blue', label='LIS r=175')
        ax.set_yscale('log')
        ax.set_ylabel('Marginal Variance')
        ax.set_title('Posterior Variance - Atmospheric Parameters')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
    

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

        ax.plot(self.truth[indX], self.truth[indY], 'ro', label='Truth', markersize=8)     
        ax.scatter(x_vals[indX,:], x_vals[indY,:], c='cornflowerblue', s=0.5)

        # plot Isofit mean/cov
        meanIsofit = np.array([self.isofitMuPos[indX], self.isofitMuPos[indY]])
        covIsofit = self.isofitGammaPos[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanIsofit[0], meanIsofit[1], 'kx', label='MAP/Laplace', markersize=12)
        self.drawEllipse(meanIsofit, covIsofit, ax, colour='black')
        
        # plot MCMC mean/cov
        meanMCMC = np.array([self.MCMCmean[indX], self.MCMCmean[indY]])
        covMCMC = self.MCMCcov[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanMCMC[0], meanMCMC[1], 'bx', label='MCMC', markersize=12)
        self.drawEllipse(meanMCMC, covMCMC, ax, colour='blue')
        
        # ax.set_title('MCMC - Two Component Visual')
        # ax.set_xlabel('Index ' + str(indX))
        # ax.set_ylabel('Index ' + str(indY))
        # ax.legend()
        return ax



    

    










