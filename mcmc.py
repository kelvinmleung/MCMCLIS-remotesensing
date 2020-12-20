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
        
        self.x0 = 0
        self.sd = 0
        self.yobs = 0

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

    def initValue(self, x0, yobs, sd, Nsamp, project=False, nr=427):
        self.startX = x0

        self.x0 = x0 - self.mu_x # subtract the mean
        
        self.yobs = yobs
        self.sd = sd
        self.Nsamp = Nsamp
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
        #print(np.linalg.eig(self.noisecov))
        #print(np.linalg.eig(H))

        #H2 = self.G[self.bandsX].T @ np.linalg.inv(self.noisecov) @ self.G

        Hn = cholPr.T @ H @ cholPr
        return H, Hn
    
    def LISproject(self, nr):
        # solve eigenvalue problem using Hn
        print('Solving generalized eigenvalue problem...')
        Hbar, Hn = self.estHessian()
        
        Lpr = np.linalg.cholesky(self.gamma_x)
        #Lpr = np.linalg.cholesky(self.gamma_x[self.bandsX,:][:,self.bandsX])

        #eigval, eigvec = np.linalg.eig(Hn)
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

        V = V[:,:nr] 
        phi = Lpr @ V
        theta = np.linalg.inv(Lpr.T) @ V
        proj = phi @ theta.T
        print('Eigenvalue problem solved.')

        # initialize the new proposal covariance as well 
        #self.propcov = self.sd * np.linalg.inv(Hbar + np.linalg.inv(self.gamma_x))

        return phi, theta, proj, phiComp, thetaComp, projComp
        
    def logpos(self, x):
        
        if self.project == True:
            xr = x
            nComp = np.shape(self.phiComp)[1]
            xComp = self.proposal_chol(np.zeros(nComp), np.identity(nComp))

            x = self.phi @ xr  + self.mu_x
            
            meas = self.fm.calc_rdn(x, self.geom)

            tPrior = xr #- self.theta.T @ self.mu_x
            tPriorComp = xComp #- self.thetaComp.T @ self.mu_x
            #tLH = self.yobs - meas
            tLH = self.yobs[self.bands] - meas[self.bands]
            gammax = np.identity(np.size(xr)) 
            gammaxComp = np.identity(nComp)
            #gammaygx = self.noisecov
            gammaygx = self.noisecov[self.bands,:][:,self.bands]

            logprior = -1/2 * tPrior.dot(np.linalg.solve(gammax, tPrior))
            loglikelihood = -1/2 * tLH.dot(np.linalg.solve(gammaygx, tLH))
            #logpriorComp = -1/2 * tPriorComp.dot(np.linalg.solve(gammaxComp, tPriorComp))

            if x[425] <= 0 or x[425] > 1 or x[426] < 1 or x[426] > 4:
                print('ATM out of bound')
                #logprior = 0
                loglikelihood = -1e10
                #logpriorComp = 0
                

        else:
            x = x + self.mu_x
            meas = self.fm.calc_rdn(x, self.geom)
            tPrior = x - self.mu_x #x[self.bandsX] - self.mu_x[self.bandsX] 
            tLH = self.yobs[self.bands] - meas[self.bands]
            gammax = self.gamma_x #[self.bandsX,:][:,self.bandsX]  
            gammaygx = self.noisecov[self.bands,:][:,self.bands]

            logprior = -1/2 * tPrior.dot(np.linalg.solve(gammax, tPrior))
            loglikelihood = -1/2 * tLH.dot(np.linalg.solve(gammaygx, tLH))
            logpriorComp = 0

            if x[425] <= 0 or x[425] > 1 or x[426] < 1 or x[426] > 4:
                logprior = 0
                loglikelihood = 0
                logpriorComp = 0
        
        return logprior + loglikelihood #+ logpriorComp

    def proposal_chol(self, mean, covCholesky):
        n = mean.size
        zx = np.random.normal(0,1,size=n)
        z = mean + covCholesky @ zx
        return z

    def alpha(self, x, z):
        logposZ = self.logpos(z)
        logposX = self.logpos(x)
        if logposZ == 0:
            print('Proposal out of bounds - Discarded.')
            return 0
        ratio = logposZ - logposX
        #ratio = self.logpos(z) - self.logpos(x)
        return np.minimum(1, np.exp(ratio))

    def runMCMC(self, alg='vanilla'):
        gammapropChol = np.linalg.cholesky(self.propcov)
        
        x_vals = np.zeros([self.x0.size, self.Nsamp])
        logpos = np.zeros(self.Nsamp)
        x = self.x0

        for i in range(self.Nsamp):
            z = self.proposal_chol(x, gammapropChol)
            #z[425:] = self.truth[425:] - self.mu_x[425:] # #######
            
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

            if (i+1) % 100 == 0: 
                print('Sample: ', i+1)
                print('\t', alpha)
                

            # change proposal covariance
            if alg == 'adaptive':
                t0 = 1000
                eps = 1e-10
                if i > t0 and i % 500 == 0: 
                    print('- New Proposal Covariance -')
                    covX = np.cov(x_vals[:,i-1000:i])
                    self.propcov = self.sd * covX + eps * np.identity(len(x))
                    if np.all(np.linalg.eigvals(self.propcov) > 0):
                        gammapropChol = np.linalg.cholesky(self.propcov)
                    else:
                        print('Proposal covariance not SPD')

        if self.project == True:
            # add samples of xComp to chain, project back to full subspace
            x_vals_full = np.zeros([self.nx, self.Nsamp])
            nComp = np.shape(self.phiComp)[1]
            for i in range(self.Nsamp):
                xComp = self.proposal_chol(np.zeros(nComp), np.identity(nComp))
                x_vals_full[:,i] = self.phi @ x_vals[:,i] + self.phiComp @ xComp + self.mu_x
                #x_vals_full[:,i] = self.phi @ x_vals[:,i] + self.mu_x #+self.phiComp @ xComp 
                #x_vals_full[self.bandsX,i] = self.phi @ x_vals[:,i] + self.phiComp @ xComp + self.mu_x
            np.save('results/MCMC/MCMC_x.npy', x_vals_full)
        else:
            x_vals_full = np.zeros([self.nx, self.Nsamp])
            for i in range(self.Nsamp):
                x_vals_full[:,i] = x_vals[:,i] + self.mu_x
            np.save('results/MCMC/MCMC_x.npy', x_vals_full)
            
        np.save('results/MCMC/logpos.npy', logpos)
        return x_vals   

    def twoDimVisualPosDensity(self, indX, indY, rfl):

        # make contour lines
        X_plot = np.arange(-0.3,0.7,0.05)
        Y_plot = np.arange(0,1,0.05)

        X_plot= np.arange(-2,2,0.05)
        Y_plot = np.arange(-2,2,0.05)
        density = np.zeros([X_plot.shape[0], Y_plot.shape[0]])
        for i in range(X_plot.shape[0]):
            for j in range(Y_plot.shape[0]):
                x = self.x0
                x[indX] = X_plot[i]
                x[indY] = Y_plot[j]
                density[i,j] = self.logpos(x)
        X_plot, Y_plot = np.meshgrid(X_plot, Y_plot)

        # plot contour
        x_vals = np.load('results/MCMC/MCMC_x.npy')
        x_mean = np.mean(x_vals, axis=1)
        fig, ax = plt.subplots()
        CS = ax.contour(X_plot, Y_plot, density)
        ax.clabel(CS, inline=1, fontsize=10)

        # plot accepted MCMC points
        #ax.plot(self.x0[indX], self.x0[indY], 'o', label='initial value', markersize=6)
        #ax.plot(self.mu_x[indX], self.mu_x[indY], 'rx', label='prior mean',markersize=12)
        #ax.plot(x_mean[indX], x_mean[indY], 'go', label='MCMC mean',markersize=10)
        #ax.scatter(x_vals[indX,:], x_vals[indY,:], s=0.5)
        ax.plot(rfl[indX], rfl[indY], 'go', label='Isofit Pos. Mean',markersize=10 )
        ax.set_title('Log Posterior Density, MCMC')
        ax.set_xlabel('Index ' + str(indX))
        ax.set_ylabel('Index ' + str(indY))
        #ax.set_xlim(left=0)
        #ax.set_ylim(bottom=0)
        ax.legend()    
        
    def twoDimVisual(self, indX, indY, t0):

        rfl = self.mupos_isofit
        #x0 = self.x0
        x_vals = np.load('results/MCMC/MCMC_x.npy')
        
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
        ellipse = Ellipse((0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor='None',
            edgecolor='red')

        scale_x = np.sqrt(cov[0, 0]) * 1
        mean_x = rfl[indX]
        scale_y = np.sqrt(cov[1, 1]) * 1 
        mean_y = rfl[indY]

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
        
        ax.set_title('MCMC - Two Component Visual')
        ax.set_xlabel('Index ' + str(indX))
        ax.set_ylabel('Index ' + str(indY))
        ax.legend()

    def calcMeanCov(self, N):
        x_vals = np.load('results/MCMC/MCMC_x.npy')
        x_ref = x_vals[:,-1*N:]
        nx = x_ref.shape[0]
        mean = np.mean(x_ref, axis=1)
        cov = np.cov(x_ref)
        return mean, cov

    def autocorr(self, ind):
        x_vals = np.load('results/MCMC/MCMC_x.npy')
        x_elem = x_vals[ind,:]

        Nsamp = 10000#len(x_elem)
        meanX = np.mean(x_elem)
        varX = np.var(x_elem)
        ac = np.zeros(Nsamp-1)

        for k in range(Nsamp-1):
            for i in range(Nsamp - k):
                ac[k] = ac[k] + 1/(Nsamp-1) * (x_elem[i] - meanX) * (x_elem[i+k] - meanX)
        ac = ac / varX

        return ac

    def plotElement(self, ind):
        x_vals = np.load('results/MCMC/MCMC_x.npy')
        x_elem = x_vals[ind,:]

        plt.figure(ind)
        plt.plot(range(self.Nsamp), x_elem, linewidth=0.5, label='Elem '+str(elem))

        plt.title('MCMC')
        plt.xlabel('Sample')
        plt.ylabel('Reflectance')
        plt.legend()
        plt.grid()
        
    def plotMCMCmean(self, mean, fig, mcmcType='pos'):
        plt.figure(fig)

        #if self.project == True:
        #    mean = self.phi @ mean
        self.setup.plotbands(mean[:425], 'g', label='MCMC Mean')
        self.setup.plotbands(self.mupos_isofit[:425], 'k', label='Isofit Pos. Mean')
        self.setup.plotbands(self.truth[:425], 'r', label='True Reflectance')
        plt.title('MCMC Results, '+ str(self.Nsamp)+' Samples')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.legend()
        plt.grid()
        
        logpos = np.load('results/MCMC/logpos.npy')
        plt.figure(fig+1)
        plt.plot(range(self.Nsamp), logpos)
        plt.xlabel('Sample')
        plt.ylabel('Log Posterior')
        plt.title('Log Posterior Plot')
        plt.grid()

    def diagnostics(self, indSet):
        # assume there are 10 elements in indSet
        # default: indSet = [10,20,50,100,150,160,250,260,425,426]
        x_vals = np.load('results/MCMC/MCMC_x.npy')

        self.twoDimVisual(indX=indSet[0], indY=indSet[1], t0=0)
        self.twoDimVisual(indX=indSet[2], indY=indSet[3], t0=0)
        self.twoDimVisual(indX=indSet[4], indY=indSet[5], t0=0)
        self.twoDimVisual(indX=indSet[6], indY=indSet[7], t0=0)
        self.twoDimVisual(indX=indSet[8], indY=indSet[9], t0=0)

        xPlot = [0,0,1,1,2,2,3,3,4,4]
        yPlot = [0,1,0,1,0,1,0,1,0,1]
        samp = range(self.Nsamp)
        sampAC = range(1,10000)
        plt.figure(18)
        fig1, axs1 = plt.subplots(5, 2)
        plt.figure(19)
        fig2, axs2 = plt.subplots(5, 2)
        for i in range(len(indSet)):
            print('Diagnostics:',indSet[i])
            x_elem = x_vals[i,:]
            xp = xPlot[i]
            yp = yPlot[i]

            axs1[xp,yp].plot(samp, x_elem)
            axs1[xp,yp].set_title('Trace - Index ' + str(indSet[i]))

            ac = self.autocorr(indSet[i])
            axs2[xp,yp].plot(sampAC, ac)
            axs2[xp,yp].set_title('Autocorrelation - Index ' + str(indSet[i]))

        

        return
        
        

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