import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms



class MCMC:
    '''
    Contains functions to perform MCMC sampling
    Integrating LIS capabilities
    '''

    def __init__(self, setup, analysis):

        self.mcmcDir = setup.mcmcDir
        
        self.x0 = 0
        self.sd = 0
        self.yobs = 0
        self.burn = 0

        # isofit parameters and functions
        self.wavelengths = setup.wavelengths
        self.reflectance = setup.reflectance
        self.truth = setup.truth
        self.radiance = setup.radiance
        self.bands = setup.bands
        self.bandsX = setup.bandsX

        self.setup = setup
        self.fm = setup.fm
        self.geom = setup.geom
        self.mu_x, self.gamma_x = setup.getPrior()
        self.mupos_isofit = setup.isofitMuPos
        self.gammapos_isofit = setup.isofitGammaPos
        self.noisecov = setup.noisecov
        
        self.gamma_ygx = analysis.gamma_ygx 
        self.G = analysis.phi

        self.nx = self.gamma_x.shape[0]
        self.ny = self.noisecov.shape[0]

    def initValue(self, x0, yobs, sd, Nsamp, burn, project=False, nr=427):
        self.startX = x0

        self.x0 = x0 - self.mu_x # subtract the mean
        
        self.yobs = yobs
        self.sd = sd
        self.Nsamp = Nsamp
        self.burn = burn
        self.project = project
        self.propcov = self.gammapos_isofit * sd

        if project == True:
            self.phi, self.theta, self.proj, self.phiComp, self.thetaComp, self.projComp = self.LISproject(nr)
            self.x0 = self.theta.T @ self.x0
            self.propcov = self.theta.T @ self.propcov @ self.theta

            self.nr = nr
            
        
    def estHessian(self):
        cholPr = np.linalg.cholesky(self.gamma_x)
        H = self.G.T @ np.linalg.inv(self.noisecov) @ self.G
        Hn = cholPr.T @ H @ cholPr
        return Hn
    
    def LISproject(self, nr):
        # solve eigenvalue problem using Hn
        print('Solving generalized eigenvalue problem...')
        Hn = self.estHessian()
        
        Lpr = np.linalg.cholesky(self.gamma_x)

        eigvec, eigval, p = np.linalg.svd(Hn)
        idx = eigval.argsort()[::-1]
        eigval = np.real(eigval[idx])
        V = np.real(eigvec[:,idx])

        # plt.semilogy(eigval)
        # plt.title('LIS Eigenvalue Decay')
        # plt.grid()
        # plt.show()

        # complementary subspace
        VComp = V[:,nr:]
        phiComp = Lpr @ VComp
        thetaComp = np.linalg.inv(Lpr.T) @ VComp
        projComp = phiComp @ thetaComp.T

        # LIS subspace
        V = V[:,:nr] 
        phi = Lpr @ V
        theta = np.linalg.inv(Lpr.T) @ V
        proj = phi @ theta.T
        print('Eigenvalue problem solved.')

        return phi, theta, proj, phiComp, thetaComp, projComp
        
    def logpos(self, x):
        
        if self.project == True:
            xr = x

            x = self.phi @ xr  + self.mu_x
            gammax = np.identity(np.size(xr)) 
            tPrior = xr #- self.theta.T @ self.mu_x
            logprior = -1/2 * tPrior.dot(np.linalg.solve(gammax, tPrior))
            
            meas = self.fm.calc_rdn(x, self.geom)
            tLH = self.yobs[self.bands] - meas[self.bands]
            gammaygx = self.noisecov[self.bands,:][:,self.bands]
            loglikelihood = -1/2 * tLH.dot(np.linalg.solve(gammaygx, tLH))
            
        else:
            x = x + self.mu_x

            tPrior = x - self.mu_x 
            logprior = -1/2 * tPrior.dot(np.linalg.solve(self.gamma_x, tPrior))

            meas = self.fm.calc_rdn(x, self.geom)
            tLH = self.yobs[self.bands] - meas[self.bands]
            gammaygx = self.noisecov[self.bands,:][:,self.bands]
            loglikelihood = -1/2 * tLH.dot(np.linalg.solve(gammaygx, tLH))

        if x[425] <= 0 or x[426] <= 1:
                print('ATM parameter is negative')
                loglikelihood = -np.Inf
        
        return logprior + loglikelihood 

    def proposal_chol(self, mean, covCholesky):
        n = mean.size
        zx = np.random.normal(0,1,size=n)
        z = mean + covCholesky @ zx
        return z

    def alpha(self, x, z):
        ratio = self.logpos(z) - self.logpos(x)
        return np.minimum(1, np.exp(ratio))

    def runMCMC(self, alg):
        gammapropChol = np.linalg.cholesky(self.propcov)
        
        x_vals = np.zeros([self.x0.size, self.Nsamp])
        logpos = np.zeros(self.Nsamp)
        x = self.x0

        for i in range(self.Nsamp):
            z = self.proposal_chol(x, gammapropChol)
            # fix the atm parameters to a constant
            # z[425:] = self.truth[425:] - self.mu_x[425:]
            
            # plot the proposal
            # if self.project == True:
            #     plt.plot(self.phi @ z + self.mu_x)
            #     plt.ylim([-0.1, 0.8])
            #     plt.show(block=False)
            #     plt.pause(0.001)
            #     plt.close()
            
            alpha = self.alpha(x, z)
            if np.random.random() < alpha:
                x = z 
            x_vals[:,i] = x
            logpos[i] = self.logpos(x)

            # print progress
            if (i+1) % 100 == 0: 
                print('Sample: ', i+1)
                print('\t', alpha)
                
            # change proposal covariance
            if alg == 'adaptive':
                eps = 1e-10
                if i > 1000 and i % 500 == 0: 
                    print('- New Proposal Covariance -')
                    covX = np.cov(x_vals[:,i-1000:i])
                    self.propcov = self.sd * covX + eps * np.identity(len(x))
                    gammapropChol = np.linalg.cholesky(self.propcov)
                    # if np.all(np.linalg.eigvals(self.propcov) > 0):
                    #     gammapropChol = np.linalg.cholesky(self.propcov)
                    # else:
                    #     print('Proposal covariance not SPD')
        
        # post processing, store MCMC chain
        x_vals_full = np.zeros([self.nx, self.Nsamp])
        if self.project == True:
            # add samples of xComp to chain, project back to full subspace
            nComp = np.shape(self.phiComp)[1]
            for i in range(self.Nsamp):
                xComp = self.proposal_chol(np.zeros(nComp), np.identity(nComp))
                x_vals_full[:,i] = self.phi @ x_vals[:,i] + self.phiComp @ xComp + self.mu_x
        else:
            for i in range(self.Nsamp):
                x_vals_full[:,i] = x_vals[:,i] + self.mu_x

        np.save(self.mcmcDir + 'MCMC_x.npy', x_vals_full)
        np.save(self.mcmcDir + 'logpos.npy', logpos)
        return x_vals   
        
    def twoDimVisual(self, indX, indY, t0):

        rfl = self.mupos_isofit
        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy')
        
        x_mean = np.mean(x_vals, axis=1)
        fig, ax = plt.subplots()
        ax.plot(self.truth[indX], self.truth[indY], 'bo', label='True reflectance', markersize=6)
        ax.plot(rfl[indX], rfl[indY], 'rx', label='Isofit pos. mean',markersize=12)
        ax.plot(x_mean[indX], x_mean[indY], 'go', label='MCMC mean',markersize=10)
        ax.scatter(x_vals[indX,t0:], x_vals[indY,t0:], s=0.5)

        covX = self.gammapos_isofit[indX, indX]
        covY = self.gammapos_isofit[indY, indY]
        covXY = self.gammapos_isofit[indX, indY]
        cov = np.array([[covX, covXY], [covXY, covY]])
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset. 
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='None', edgecolor='red')

        scale_x = np.sqrt(cov[0, 0]) * 1
        mean_x = rfl[indX]
        scale_y = np.sqrt(cov[1, 1]) * 1 
        mean_y = rfl[indY]

        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
        
        ax.set_title('MCMC - Two Component Visual')
        ax.set_xlabel('Index ' + str(indX))
        ax.set_ylabel('Index ' + str(indY))
        ax.legend()

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

        Nsamp = min(self.Nsamp, 10000)
        meanX = np.mean(x_elem)
        varX = np.var(x_elem)
        ac = np.zeros(Nsamp-1)

        for k in range(Nsamp-1):
            for i in range(Nsamp - k):
                ac[k] = ac[k] + 1/(Nsamp-1) * (x_elem[i] - meanX) * (x_elem[i+k] - meanX)
        ac = ac / varX

        return ac

    def plotElement(self, ind):
        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy')
        x_elem = x_vals[ind,:]

        plt.figure(ind)
        plt.plot(range(self.Nsamp), x_elem, linewidth=0.5, label='Elem '+str(elem))

        plt.title('MCMC')
        plt.xlabel('Sample')
        plt.ylabel('Reflectance')
        plt.legend()
        plt.grid()
        
    def plotMCMCmean(self, mean, fig):
        plt.figure(fig)

        self.setup.plotbands(mean[:425], 'g', label='MCMC Mean')
        self.setup.plotbands(self.mupos_isofit[:425], 'k', label='Isofit Pos. Mean')
        self.setup.plotbands(self.truth[:425], 'r', label='True Reflectance')
        plt.title('MCMC Results, '+ str(self.Nsamp)+' Samples')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.legend()
        plt.grid()
        
        logpos = np.load(self.mcmcDir + 'logpos.npy')
        plt.figure(fig+1)
        plt.plot(range(self.Nsamp), logpos)
        plt.xlabel('Sample')
        plt.ylabel('Log Posterior')
        plt.title('Log Posterior Plot')
        plt.grid()

    def diagnostics(self, indSet=[10,20,50,100,150,160,250,260,425,426], calcAC=False):
        # assume there are 10 elements in indSet
        # default: indSet = [10,20,50,100,150,160,250,260,425,426]
        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy')

        self.twoDimVisual(indX=indSet[0], indY=indSet[1], t0=0)
        self.twoDimVisual(indX=indSet[2], indY=indSet[3], t0=0)
        self.twoDimVisual(indX=indSet[4], indY=indSet[5], t0=0)
        self.twoDimVisual(indX=indSet[6], indY=indSet[7], t0=0)
        self.twoDimVisual(indX=indSet[8], indY=indSet[9], t0=0)

        xPlot = [0,0,1,1,2,2,3,3,4,4]
        yPlot = [0,1,0,1,0,1,0,1,0,1]
        fig1, axs1 = plt.subplots(5, 2)
        if calcAC == True:
            fig2, axs2 = plt.subplots(5, 2)
        for i in range(len(indSet)):
            print('Diagnostics:',indSet[i])
            x_elem = x_vals[i,:]
            xp = xPlot[i]
            yp = yPlot[i]

            axs1[xp,yp].plot(range(self.Nsamp), x_elem)
            axs1[xp,yp].set_title('Trace - Index ' + str(indSet[i]))

            if calcAC == True:
                ac = self.autocorr(indSet[i])
                axs2[xp,yp].plot(range(1,len(ac)+1), ac)
                axs2[xp,yp].set_title('Autocorrelation - Index ' + str(indSet[i]))
        

    def maxLogPos(self, N):
        maxlogpos = -1e10
        cholCov = np.linalg.cholesky(self.gamma_x) / 2
        logPos = np.zeros(N)
        dist = np.zeros([N,self.nx])
        for i in range(N):
            if (i+1) % 100 == 0:
                print('Iteration:', i+1)
            while logPos[i] == 0:
                x = self.proposal_chol(self.truth, cholCov)
                dist[i,:] = x
                
                logPos[i] = self.logpos(x - self.mu_x)
        indMax = np.argmax(logPos)

        plt.figure(1)
        self.setup.plotbands(self.mu_x, 'k', label='Prior')
        self.setup.plotbands(self.truth, 'b', label='Truth')
        self.setup.plotbands(self.mupos_isofit, 'r', label='Isofit Posterior')
        self.setup.plotbands(dist[indMax,:], 'g.', label='Max Log Pos')
        plt.legend()
        plt.grid()

        return logPos[indMax], self.logpos(self.truth), dist[indMax,:]