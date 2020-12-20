import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale, StandardScaler

class Regression:
    '''
    Contains functions to perform Lasso Regression to 
    create a linear approximation of the Isofit forward model.
    '''

    def __init__(self, setup):
        
        # setup parameters
        self.plotbands = setup.plotbands
        self.wavelengths = setup.wavelengths
        self.reflectance = setup.reflectance
        self.truth = setup.truth
        self.radiance = setup.radiance
        self.bands = setup.bands
        self.bandsX = setup.bandsX

        # load data sets
        X_train = np.load('results/samples/X_train.npy')
        Y_train = np.load('results/samples/Y_train.npy')
        X_test = np.load('results/samples/X_test.npy')
        Y_test = np.load('results/samples/Y_test.npy')

        # scale the data
        self.scalerX = StandardScaler().fit(X_train)
        self.scalerY = StandardScaler().fit(Y_train)

        self.X_train = self.scalerX.transform(X_train)
        self.Y_train = self.scalerY.transform(Y_train)
        self.X_test = self.scalerX.transform(X_test)
        self.Y_test = self.scalerY.transform(Y_test)

        # get mean and variance for future transformations
        self.meanX = self.scalerX.mean_
        self.varX = self.scalerX.var_

        self.meanY = self.scalerY.mean_
        self.varY = self.scalerY.var_

        N = self.X_train.shape[0]
        
        
        nx = self.reflectance.size
    
        self.reflectance_scaled = (self.reflectance - self.meanX[:nx]) / np.sqrt(self.varX[:nx])
        self.truth_scaled = (self.truth - self.meanX) / np.sqrt(self.varX)
        
        
    def scaleDownX(self, X):
        if X.ndim == 1:
            return (X - self.meanX) / np.sqrt(self.varX)
        else:
            return self.scalerX.transform(X)

    def scaleDownY(self, Y):
        if Y.ndim == 1:
            return (Y - self.meanY) / np.sqrt(self.varY)
        else:
            return self.scalerY.transform(Y)
    '''
    def reglasso(self, param, yElem):

        linreg = Lasso(alpha=param, max_iter=5000) 
        linreg.fit(self.X_train, self.Y_train[:,yElem])
        phi = linreg.coef_
        pred = linreg.predict(self.X_train)
        trainError = mean_squared_error(self.Y_train[:,yElem], pred)

        N = self.X_test.shape[0]
        genError = 0
        for i in range(N):
            genError = genError + 1/N * np.linalg.norm(self.Y_test[i,yElem] - phi.dot(self.X_test[i,:]))
            
        return phi, trainError, genError
    '''
    def reglasso(self, param, yElem):
        # channel removal
        #X_train = self.X_train[:,self.bandsX]
        # don't remove channels
        X_train = self.X_train

        # train with the reduced X_train
        linreg = Lasso(alpha=param, max_iter=5000) 
        linreg.fit(X_train, self.Y_train[:,yElem])
        phiReduce = linreg.coef_
        pred = linreg.predict(X_train)
        trainError = mean_squared_error(self.Y_train[:,yElem], pred)

        # make phi back into 427
        # nx = self.X_train.shape[1]
        # phi = np.zeros(nx)
        # j = 0
        # for i in range(nx):
        #     if i in self.bandsX:
        #         phi[i] = phiReduce[j] 
        #         j = j + 1
        phi = phiReduce
        
        N = self.X_test.shape[0]
        genError = 0
        for i in range(N):
            genError = genError + 1/N * np.linalg.norm(self.Y_test[i,yElem] - phi.dot(self.X_test[i,:]))
        return phi, trainError, genError

    def plotlog(self, plotx, ploty, fig, title='', xLabel='', yLabel=''):
        plt.figure(fig)
        plt.loglog(plotx, ploty, linewidth=2)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()
        plt.grid()
        return

    def tuneLasso(self, params, yElem, plot=True):
        nParam = np.array(params).size
        trainError = np.zeros(nParam)
        genError = np.zeros(nParam)

        for j in range(nParam):
            print('Regression parameter:', params[j])
            [phi, te, ge] = self.reglasso(params[j], yElem)
            trainError[j] = te
            genError[j] = ge
            print('\t Gen Error:', ge, '\t Train Error:', te)
        if plot==True:
            self.plotlog(params,trainError, 11, title='Lasso - Training Error', xLabel='Regularization Parameter', yLabel='Error')
            self.plotlog(params,genError,12, title='Lasso - Generalization Error', xLabel='Regularization Parameter', yLabel='Error')

        indMin = np.argmin(genError)
        return trainError, genError, params[indMin]

    def fullLasso(self, params):
        # perform lasso for all wavelengths
        # change phi to ny by nx
        nx = self.X_train.shape[1]
        ny = self.Y_train.shape[1]

        y_lasso = np.zeros(ny)
        GE = np.zeros(ny)
        TE = np.zeros(ny)
        phi = np.zeros([ny,nx])
        '''
        for i in range(ny):
            print('Regression: Element', i+1, str(self.wavelengths[i]) + 'nm')
            phi[:,i], te, ge = self.reglasso(params[i], i)
            scaled = phi[:,i].dot(self.truth_scaled) #self.reflectance_scaled
            y_lasso[i] = scaled * np.sqrt(self.varY[i]) + self.meanY[i]
            TE[i] = te
            GE[i] = ge
        '''
        for i in self.bands:
            print('Regression: Element', i+1, str(self.wavelengths[i]) + 'nm')
            phi[i,:], te, ge = self.reglasso(params[i], i)
            scaled = phi[i,:].dot(self.truth_scaled) 
            y_lasso[i] = scaled * np.sqrt(self.varY[i]) + self.meanY[i]
            TE[i] = te
            GE[i] = ge

        np.save('results/Regression/y_lasso.npy', y_lasso)
        np.save('results/Regression/lassoTE.npy', TE)
        np.save('results/Regression/lassoGE.npy', GE)
        np.save('results/Regression/phi.npy',phi)      

    def plotFullLasso(self):

        y_lasso = np.load('results/Regression/y_lasso.npy')
        TE = np.load('results/Regression/lassoTE.npy')
        GE = np.load('results/Regression/lassoGE.npy')
        phi = np.load('results/Regression/phi.npy')

        plt.figure(31)
        plt.plot(self.wavelengths, self.radiance, 'navy',linewidth=2, label='Isofit Forward Model')
        plt.plot(self.wavelengths, y_lasso, 'orange',linewidth=1, label='Linear Model')
        #self.plotbands(self.radiance,'navy',linewidth=2, label='Isofit Forward Model')
        #self.plotbands(y_lasso, 'orange', linewidth=1, label='Linear Model')
        #plt.ylim([-2,20])
        plt.xlabel('Wavelength')
        plt.ylabel('Radiance')
        plt.title('Lasso Regression - Radiance')
        plt.grid()
        plt.legend()

        plt.figure(34)
        self.plotbands(GE, 'navy', linewidth=1, label='Generalization Error', axis='semilogy')
        self.plotbands(TE, 'orange', linewidth=1, label='Training Error', axis='semilogy')
        #plt.semilogy(self.wavelengths, GE, linewidth=1, label='Generalization Error')
        #plt.semilogy(self.wavelengths, TE, linewidth=1, label='Training Error')
        plt.xlabel('Wavelength')
        plt.ylabel('Error')
        plt.title('Lasso Regression - Error')
        plt.grid()
        plt.legend()

        relerror = abs(self.radiance - y_lasso) / abs(self.radiance)
        plt.figure(35)
        self.plotbands(relerror, 'navy',linewidth=1, label='Lasso,p=1e-3',axis='semilogy')
        #plt.semilogy(self.wavelengths, relerror, linewidth=1, label='Lasso,p=1e-3')
        plt.xlabel('Wavelength')
        plt.ylabel('Error')
        plt.title('Lasso Regression - Relative Error')
        plt.grid()
        plt.legend()

        plt.figure(36)
        plt.spy(phi, precision=1e-15,markersize=2)
        plt.title(r'Sparsity Plot - $\Phi$')


    def plotRegSample(self, fig, yElem, phi):

        # plot the training + test data, plus the line
        plt.figure(fig)
        plt.plot(self.X_train[:,yElem], self.Y_train[:,yElem],'bx',label='Training Data')
        plt.plot(self.X_test[:,yElem], self.Y_test[:,yElem],'rx', label='Test Data')
        linearApprox = np.zeros(10000)
        for i in range(10000):
            linearApprox[i] = phi.dot(self.X_train[i,:])

        polynomial = np.polyfit(self.X_train[:,yElem], linearApprox, 1)
        xLine = np.linspace(-5,5,50)
        plt.plot(xLine, np.poly1d(polynomial)(xLine))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Lasso Regression at '+str(round(self.wavelengths[yElem]))+'nm, alpha=1e-3')
        plt.grid()
        plt.legend()
        plt.xlim([1,2])
        plt.ylim([1,2])