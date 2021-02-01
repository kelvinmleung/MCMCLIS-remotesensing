import subprocess

remoteDir = "kmleung@hypersonic.mit.edu:/master/home/kmleung/JPLproject/results/MCMC/"
localDir = "/Users/KelvinLeung/Documents/JPLproject/results/MCMC/fromCluster"

indSet = [30,40,90,100,150,160,250,260,425,426]

listFiles = []
listFiles = listFiles + ['reflMean.png', 'atmMean.png', 'reflVar.png', 'atmVar.png', ' errorRelCov.png', 'trace.png', 'autocorr.png', 'logpos.png']
numPairs = int(len(indSet) / 2)
for i in range(numPairs):
    listFiles = listFiles + ['2D_' + str(indSet[2*i]) + '-' + str(indSet[2*i+1]) + '.png']
for i in range(len(listFiles)):
    subprocess.call(["rsync", "-chavzP", "--stats", "--delete", remoteDir + listFiles[i], localDir])



