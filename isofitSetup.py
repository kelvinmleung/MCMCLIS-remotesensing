import sys, os, json
import numpy as np
import scipy as s
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib

import scipy.stats as st

sys.path.insert(0, '../isofit/')

import isofit
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.inversion.inverse import Inversion
from isofit.configs.configs import Config  
from isofit.surface.surface_multicomp import MultiComponentSurface
    
class Setup:
    '''
    Contains functions to generate training and test samples
    from isofit.
    '''
    def __init__(self, wv, ref, atm, mcmcdir='MCMCRun', rad='simulated'):

        print('Setup in progress...')
        self.wavelengths = wv
        self.reflectance = ref
        self.truth = np.concatenate((ref, atm))
        # np.save('x0isofit/atmSample.npy', [atm[0], atm[1]]) # set for Isofit initialization

        # specify storage directories 
        self.sampleDir = '../results/Regression/samples/'
        self.regDir = '../results/Regression/linearModel/'
        self.analysisDir = '../results/Analysis/'
        self.mcmcDir = '../results/MCMC/' + mcmcdir + '/'

        # load Isofit
        with open('setup/config/config_inversion.json', 'r') as f:
            config = json.load(f)
        fullconfig = Config(config)
        self.fm = ForwardModel(fullconfig)
        self.geom = Geometry()
        self.mu_x, self.gamma_x = self.getPrior(fullconfig)

        # get Isofit noise model and simulate radiance
        if rad =='simulated':
            rad = self.fm.calc_rdn(self.truth, self.geom)
            self.noisecov = self.fm.Seps(self.truth, rad, self.geom)
            eps = np.random.multivariate_normal(np.zeros(len(rad)), self.noisecov)
            self.radiance = rad
            self.radNoisy = rad + eps
        # elif rad == 'real':
        #     self.radiance = 


        
        # inversion using simulated radiance
        self.isofitMuPos, self.isofitGammaPos = self.invModel(self.radNoisy)
        self.nx = self.truth.shape[0]
        self.ny = self.reflectance.shape[0]
        
        # get indices that are in the window (i.e. take out deep water spectra)
        wl = self.wavelengths
        bands = []
        for i in range(wl.size):
            if (wl[i] > 380 and wl[i] < 1300) or (wl[i] > 1450 and wl[i] < 1780) or (wl[i] > 1950 and wl[i] < 2450):
                bands = bands + [i]
        self.bands = bands
        self.bandsX = bands + [425,426]

    def getPrior(self, fullconfig):
        # get index of prior used in inversion
        mcs = MultiComponentSurface(fullconfig)
        indPr = mcs.component(self.truth, self.geom)
        print('Prior Index:', indPr)
        # Get prior mean and covariance
        surfmat = loadmat('setup/data/surface.mat')
        wl = surfmat['wl'][0]
        refwl = np.squeeze(surfmat['refwl'])
        idx_ref = [np.argmin(abs(wl-w)) for w in np.squeeze(refwl)]
        idx_ref = np.array(idx_ref)
        refnorm = np.linalg.norm(self.reflectance[idx_ref])

        mu_priorsurf = self.fm.surface.components[indPr][0] * refnorm
        mu_priorRT = self.fm.RT.xa()
        mu_priorinst = self.fm.instrument.xa()
        mu_x = np.concatenate((mu_priorsurf, mu_priorRT, mu_priorinst), axis=0)
        
        gamma_priorsurf = self.fm.surface.components[indPr][1] * (refnorm ** 2)
        gamma_priorRT = self.fm.RT.Sa()[:, :]
        gamma_priorinst = self.fm.instrument.Sa()[:, :]
        gamma_x = s.linalg.block_diag(gamma_priorsurf, gamma_priorRT, gamma_priorinst)

        return mu_x, gamma_x

    def invModel(self, radiance):

        inversion_settings = {"implementation": {
        "mode": "inversion",
        "inversion": {
        "windows": [[380.0, 1300.0], [1450, 1780.0], [1950.0, 2450.0]]}}}
        inverse_config = Config(inversion_settings)
        iv = Inversion(inverse_config, self.fm)
        state_trajectory = iv.invert(radiance, self.geom)
        state_est = state_trajectory[-1]
        rfl_est, rdn_est, path_est, S_hat, K, G = iv.forward_uncertainty(state_est, radiance, self.geom)

        return state_est, S_hat

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
        plt.contourf(X_plot, Y_plot, gamma)
        plt.title('Covariance')
        plt.axis('equal')
        plt.colorbar()

    def genErrorMAP(self, numSamp):

        deltaTwo = abs(self.truth - self.isofitMuPos) * 2

        invPrior = np.linalg.inv(self.gamma_x)
        invNoise = np.linalg.inv(self.noisecov)
        varIsofit = np.diag(self.isofitGammaPos)
        varPrior = np.diag(self.gamma_x)
        varApprox = np.zeros([self.nx, numSamp])
        plotAOD = np.zeros(numSamp)
        plotH2O = np.zeros(numSamp)
        aodError = np.zeros(numSamp)
        h2oError = np.zeros(numSamp)

        for i in range(numSamp):
            if (i+1) % 100 == 0:
                print(i+1)
            sampX = np.random.uniform(self.isofitMuPos - deltaTwo, self.isofitMuPos + deltaTwo)
            sampX = abs(sampX)
            plotAOD[i] = sampX[425]
            plotH2O[i] = sampX[426]
            
            jac = self.fm.K(sampX, self.geom)
            posApprox = np.linalg.inv(jac.T @ invNoise @ jac + invPrior)
            varApprox[:,i] = np.diag(posApprox)

            aodError[i] = abs(varApprox[425,i]-varPrior[425]) / abs(varPrior[425])
            h2oError[i] = abs(varApprox[426,i]-varPrior[426]) / abs(varPrior[426])
            

        # self.plotHeatMap(plotAOD, plotH2O, varApprox[425,:], title='AOD variance')
        # self.plotHeatMap(plotAOD, plotH2O, varApprox[426,:], title='H2O variance')
        self.plotHeatMap(plotAOD, plotH2O, aodError, title='Relative Error in AOD variance')
        self.plotHeatMap(plotAOD, plotH2O, h2oError, title='Relative Error in H2O variance')
        
    def plotHeatMap(self, plotAOD, plotH2O, plotVal, title=''):
        plt.figure()
        plt.scatter(plotAOD, plotH2O, c=plotVal, s=20, marker='s', cmap='GnBu', norm=matplotlib.colors.LogNorm())
        plt.plot(self.truth[425], self.truth[426], 'rx', markersize=12, label='Truth')
        plt.plot(self.isofitMuPos[425], self.isofitMuPos[426], 'kx', markersize=12, label='MAP')
        plt.title(title)
        plt.xlabel('AOD - Index 425')
        plt.ylabel('H2O - Index 426')
        plt.colorbar()
        plt.legend()

    def plotPosterior(self, MCMCmean, MCMCcov): #mu_xgyLin, gamma_xgyLin,

        plt.figure()
        # self.plotbands(self.mu_x[:425], 'r', label='Prior')
        self.plotbands(self.truth[:425], 'b.',label='True Reflectance')
        self.plotbands(self.isofitMuPos[:425],'k.', label='Isofit Posterior')
        # self.plotbands(mu_xgyLin[:425], 'm.',label='Linear Posterior')
        self.plotbands(MCMCmean[:425], 'c.',label='MCMC Posterior')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.title('Posterior Mean Comparison')
        plt.grid()
        plt.legend()
        plt.savefig(self.mcmcDir + 'reflMean.png', dpi=300)

        plt.figure()
        isofitError = abs(self.isofitMuPos[:425] - self.truth[:425]) / abs(self.truth[:425])
        linError = abs(mu_xgyLin[:425] - self.truth[:425]) / abs(self.truth[:425])
        mcmcError = abs(MCMCmean[:425] - self.truth[:425]) / abs(self.truth[:425])
        self.plotbands(isofitError,'k.', label='Isofit Posterior',axis='semilogy')
        # self.plotbands(linError, 'm.',label='Linear Posterior',axis='semilogy')
        self.plotbands(mcmcError, 'c.',label='MCMC Posterior',axis='semilogy')
        plt.xlabel('Wavelength')
        plt.ylabel('Relative Error')
        plt.title('Error in Posterior Mean')
        plt.grid()
        plt.legend()
        plt.savefig(self.mcmcDir + 'reflError.png', dpi=300)

        plt.figure()
        plt.plot(self.truth[425], self.truth[426], 'bo',label='True Reflectance')
        plt.plot(self.mu_x[425], self.mu_x[426], 'r.',label='Prior')
        plt.plot(self.isofitMuPos[425],self.isofitMuPos[426],'k.', label='Isofit Posterior')
        # plt.plot(mu_xgyLin[425], mu_xgyLin[426],'mx',label='Linear Posterior')
        plt.plot(MCMCmean[425], MCMCmean[426], 'cx',label='MCMC Posterior')
        plt.xlabel('AOT550')
        plt.ylabel('H2OSTR')
        plt.grid()
        plt.legend()
        plt.savefig(self.mcmcDir + 'atmMean.png', dpi=300)

        # bar graph of atm parameter variances
        isofitErrorAtm = abs(self.isofitMuPos[425:] - self.truth[425:]) / abs(self.truth[425:])
        linErrorAtm = abs(mu_xgyLin[425:] - self.truth[425:]) / abs(self.truth[425:])
        mcmcErrorAtm = abs(MCMCmean[425:] - self.truth[425:]) / abs(self.truth[425:])
        labels = ['425 - AOD550', '426 - H2OSTR']
        x = np.arange(len(labels))  # the label locations
        width = 0.175
        fig, ax = plt.subplots()
        rects2 = ax.bar(x - width, isofitErrorAtm, width, label='Isofit Posterior')
        # rects3 = ax.bar(x, linErrorAtm, width, label='Linear Posterior')
        rects4 = ax.bar(x + width, mcmcErrorAtm, width, label='MCMC Posterior')
        ax.set_yscale('log')
        ax.set_ylabel('Relative Error')
        ax.set_title('Error in Atm Parameters')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.savefig(self.mcmcDir + 'atmError.png', dpi=300)

        # variance plot
        priorVar = np.diag(self.gamma_x)
        isofitVar = np.diag(self.isofitGammaPos)
        linearVar = np.diag(gamma_xgyLin)
        MCMCVar = np.diag(MCMCcov)
        plt.figure()
        self.plotbands(priorVar[:425], 'b.',label='Prior', axis='semilogy')
        self.plotbands(isofitVar[:425],'k.', label='Isofit Posterior', axis='semilogy')
        # self.plotbands(linearVar[:425], 'm.',label='Linear Posterior', axis='semilogy')
        self.plotbands(MCMCVar[:425], 'c.',label='MCMC Posterior', axis='semilogy')
        plt.xlabel('Wavelength')
        plt.ylabel('Variance')
        plt.title('Marginal Variance Comparison')
        plt.grid()
        plt.legend()
        plt.savefig(self.mcmcDir + 'reflVar.png', dpi=300)

        # bar graph of atm parameter variances
        labels = ['425 - AOD550', '426 - H2OSTR']
        x = np.arange(len(labels))  # the label locations
        width = 0.175
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - 3*width/2, priorVar[425:], width, label='Prior')
        rects2 = ax.bar(x - width/2, isofitVar[425:], width, label='Isofit Posterior')
        # rects3 = ax.bar(x + width/2, linearVar[425:], width, label='Linear Posterior')
        rects4 = ax.bar(x + 3*width/2, MCMCVar[425:], width, label='MCMC Posterior')
        ax.set_yscale('log')
        ax.set_ylabel('Variance')
        ax.set_title('Marginal Variance of Atm')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.savefig(self.mcmcDir + 'atmVar.png', dpi=300)

        # plot: x-axis is error in posterior mean, y-axis is error in mean weighted by covariance
        
        isofitPlotX = np.linalg.norm(self.isofitMuPos - self.truth) ** 2
        linearPlotX = np.linalg.norm(mu_xgyLin - self.truth) ** 2
        MCMCPlotX = np.linalg.norm(MCMCmean - self.truth) ** 2

        isofitPlotY = np.linalg.norm(np.diag(isofitVar ** (-0.5)) * (self.isofitMuPos - self.truth)) ** 2
        linearPlotY = np.linalg.norm(np.diag(linearVar ** (-0.5)) * (mu_xgyLin - self.truth)) ** 2
        MCMCPlotY = np.linalg.norm(np.diag(MCMCVar ** (-0.5)) * (MCMCmean - self.truth)) ** 2

        plt.figure()
        plt.loglog(isofitPlotX, isofitPlotY, 'k*', label='Isofit')
        # plt.loglog(linearPlotX, linearPlotY, 'm*', label='Linear')
        plt.loglog(MCMCPlotX, MCMCPlotY, 'c*', label='MCMC')
        plt.title('Error in Posterior')
        plt.xlabel(r'$| \mu_{pos} - \mu_{true} |_2^2$')
        plt.ylabel(r'$| diag(\Gamma_{pos}^{-1/2}) \mu_{pos} - \mu_{true} |_2^2$')
        plt.legend()
        plt.savefig(self.mcmcDir + 'errorRelCov.png', dpi=300)
    

    def testIsofitStartPt(self, N):
        randAOD = np.zeros(N)
        randH2O = np.zeros(N)

        refl = np.zeros([N,425])
        posAOD = np.zeros(N)
        posH2O = np.zeros(N)
        varAOD = np.zeros(N)
        varH2O = np.zeros(N)

        pdfVal = np.zeros(N)

        nfev = np.zeros(N)
        status = np.zeros(N)
        
        for i in range(N):
            if (i+1) % 1 == 0:
                print('Iteration ' + str(i+1))
            randAOD[i] = (1 - 0) * np.random.random() + 0
            randH2O[i] = (4 - 1) * np.random.random() + 1

            np.save('x0isofit/atmSample.npy', [randAOD[i], randH2O[i]])

            # inversion using simulated radiance
            isofitMuPos, isofitGammaPos = self.invModel(self.radiance)

            refl[i,:] = isofitMuPos[:425]
            posAOD[i] = isofitMuPos[425]
            posH2O[i] = isofitMuPos[426]
            varAOD[i] = isofitGammaPos[425,425]
            varH2O[i] = isofitGammaPos[426,426]

            # print(isofitGammaPos[self.bandsX,:][:,self.bandsX])
            # print(np.linalg.det(isofitGammaPos[self.bandsX,:][:,self.bandsX]))

            pdfVal[i] = st.multivariate_normal.pdf(x=isofitMuPos, mean=isofitMuPos, cov=isofitGammaPos)
            nfev[i] = np.load('x0isofit/nfevTmp.npy')
            status[i] = np.load('x0isofit/statusTmp.npy')

        np.save('x0isofit/randAOD.npy', randAOD)
        np.save('x0isofit/randH2O.npy', randH2O)
        np.save('x0isofit/posAOD.npy', posAOD)
        np.save('x0isofit/posH2O.npy', posH2O)
        np.save('x0isofit/refl.npy', refl)
        np.save('x0isofit/varAOD.npy', varAOD)
        np.save('x0isofit/varH2O.npy', varH2O)
        np.save('x0isofit/pdfVal.npy', pdfVal)
        np.save('x0isofit/nfev.npy', nfev)
        np.save('x0isofit/status.npy', status)

    def plotScatterIsofitTest(self, x, y, c, title):
        plt.figure()
        plt.scatter(x, y, c=c, cmap='Blues')
        plt.title(title)
        plt.xlabel('AOD')
        plt.ylabel('H2O')
        plt.colorbar()

    def testIsofitStartPtPlot(self):
        randAOD = np.load('x0isofit/randAOD.npy')
        randH2O = np.load('x0isofit/randH2O.npy')

        x = np.load('x0isofit/posAOD.npy')
        y = np.load('x0isofit/posH2O.npy')

        refl = np.load('x0isofit/refl.npy')
        varAOD = np.load('x0isofit/varAOD.npy')
        varH2O = np.load('x0isofit/varH2O.npy')
        pdfVal = np.load('x0isofit/pdfVal.npy')

        nfev = np.load('x0isofit/nfev.npy')
        status = np.load('x0isofit/status.npy')
    
        self.plotScatterIsofitTest(x, y, pdfVal, title='PDF Value of MAP')
        plt.savefig('x0isofit/pdfVal.png', dpi=300)
        self.plotScatterIsofitTest(x, y, varAOD, title='Variance of AOD')
        plt.savefig('x0isofit/varAOD.png', dpi=300)
        self.plotScatterIsofitTest(x, y, varH2O, title='Variance of H2O')
        plt.savefig('x0isofit/varH2O.png', dpi=300)
        self.plotScatterIsofitTest(x, y, nfev, title='Number of Function Evaluations')
        plt.savefig('x0isofit/nfev.png', dpi=300)
        self.plotScatterIsofitTest(x, y, status, title='Status of Convergence')
        plt.savefig('x0isofit/status.png', dpi=300)

        # plot some sample reflectances
        plt.figure()
        for i in range(10):
            self.plotbands(refl[i,:], '-')
        plt.title('Reflectances from retrievals with random initial atm')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.savefig('x0isofit/refl.png', dpi=300)

        # plot the initial samples
        plt.figure()
        plt.scatter(randAOD, randH2O)
        plt.title('Initial Values of Atmospheric Parameters')
        plt.xlabel('AOD')
        plt.ylabel('H2O')
        plt.savefig('x0isofit/x0atm.png', dpi=300)


        '''
        [xmin, xmax] = [0, 1]
        [ymin, ymax] = [1, 4]
        
        # Perform the kernel density estimate
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
        
        # Contourf plot
        cfset = ax.contourf(xx, yy, f, cmap='Blues' ) # levels=levs
        plt.clabel(cfset, fontsize='smaller')

        # # plot truth, isofit, and mcmc mean
        # meanIsofit = np.array([self.isofitMuPos[indX], self.isofitMuPos[indY]])
        # meanMCMC = np.array([self.MCMCmean[indX], self.MCMCmean[indY]])
        # ax.plot(self.truth[indX], self.truth[indY], 'go', label='Truth', markersize=8)  
        # ax.plot(meanIsofit[0], meanIsofit[1], 'rx', label='MAP', markersize=12)
        # ax.plot(meanMCMC[0], meanMCMC[1], 'kx', label='MCMC', markersize=12)
        # ax.scatter(x, y, 'b.')

        # Label plot
        # ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel('AOD')
        ax.set_ylabel('H2O')
        ax.legend()
        fig.colorbar(cfset)
        '''

    def genStartPoints(self):
        # refl = np.load('x0isofit/refl.npy')
        randAOD = np.load('x0isofit/randAOD.npy')
        randH2O = np.load('x0isofit/randH2O.npy')
        
        N = randAOD.shape[0]
        truth = np.zeros([N,427])
        # radiances = np.zeros([N,425])

        for i in range(N):
            truth[i,:425] = self.reflectance
            truth[i,425] = randAOD[i]
            truth[i,426] = randH2O[i]

            # no noise added

            # radiances[i,:] = self.fm.calc_rdn(truth[i,:], self.geom)


        np.save('x0isofit/truths.npy', truth)
        np.save('x0isofit/radiance.npy', self.radiance)
        np.save('x0isofit/noisecov.npy', self.noisecov)





