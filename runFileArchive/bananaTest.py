
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# https://arxiv.org/pdf/1305.2634.pdf

def logpos(x):
    ''' Calculate log posterior ''' 

    var = [100, 1]
    b = 0.03
    x1 = x[0]
    x2 = x[1] + b * x[0]**2 - 100 * b
    
    density = 1
    # density at [x1,x2] for normal distribution with cov
    # norm.pdf()
    density1 = norm.pdf(x1 / np.sqrt(var[0]))
    density2 = norm.pdf(x2 / np.sqrt(var[1]))

    return np.log(density1 * density2)


nx = 2
Nsamp = 10000
x0 = np.array([0,0])
x_vals = np.zeros([nx, Nsamp])
sd = 2.4 ** 2 / nx
propcov = np.identity(nx) * sd
alg = 'adaptive'

# gamma_ygx = np.identity(nx) * 1e-4


''' Run MCMC algorithm '''
x_vals = np.zeros([x0.size, Nsamp]) # store all samples
logposterior = np.zeros(Nsamp) # store the log posterior values
x = x0 ### SUBTRACT MEAN!?!?!?!

for i in range(Nsamp):
    z = np.random.multivariate_normal(x, propcov)
    
    ### Calculate acceptance ###
    ratio = logpos(z) - logpos(x)
    alpha =  np.minimum(1, np.exp(ratio))

    if np.random.random() < alpha:
        x = z 
    x_vals[:,i] = x
    logposterior[i] = logpos(x)

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
            propcov = sd * covX + eps * np.identity(len(x))

# post processing, store MCMC chain
x_vals_full = np.zeros([nx, Nsamp])
for i in range(Nsamp):
    x_vals_full[:,i] = x_vals[:,i] #+ self.mu_x

np.save('../results/MCMC/MCMC_x_banana.npy', x_vals_full)
np.save('../results/MCMC/logpos_banana.npy', logposterior)


### AUTOCORRELATION ###
meanX1 = np.mean(x_vals[0,:])
meanX2 = np.mean(x_vals[1,:])
varX1 = np.var(x_vals[0,:])
varX2 = np.var(x_vals[1,:])
ac1 = np.zeros(Nsamp-1)
ac2 = np.zeros(Nsamp-1)

for k in range(Nsamp-1):
    for i in range(Nsamp - k):
        ac1[k] = ac1[k] + 1/(Nsamp-1) * (x_vals[0,i] - meanX1) * (x_vals[0,i+k] - meanX1)
        ac2[k] = ac2[k] + 1/(Nsamp-1) * (x_vals[1,i] - meanX2) * (x_vals[1,i+k] - meanX2)
ac1 = ac1 / varX1
ac2 = ac2 / varX2
fig2, axs2 = plt.subplots(2, 1)
axs2[0].plot(range(1,len(ac1)+1), ac1)
axs2[0].set_title('Autocorrelation - Index 0')
axs2[1].plot(range(1,len(ac2)+1), ac2)
axs2[1].set_title('Autocorrelation - Index 1')

### TRACE ###
fig1, axs1 = plt.subplots(2, 1)
axs1[0].plot(range(Nsamp), x_vals[0,:])
axs1[0].set_title('Trace - Index 0')
axs1[1].plot(range(Nsamp), x_vals[1,:])
axs1[1].set_title('Trace - Index 1')

### TWO-COMPONENT VISUAL ###
indX = 0
indY = 1
t0 = 0
#x_mean = np.mean(x_vals_full, axis=1)
fig, ax = plt.subplots()
#ax.plot(x_mean[indX], x_mean[indY], 'go', label='MCMC mean',markersize=10)
ax.scatter(x_vals[indX,t0:], x_vals[indY,t0:], s=0.5)
ax.set_title('Two Component Visual')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()



plt.show()


