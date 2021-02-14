import subprocess

''' Run this on cluster first '''
# tar -zcvf A9.tgz atmMean.png acceptance.png 2D_30-40.png 2D_90-100.png 2D_150-160.png 2D_250-260.png 2D_425-426.png autocorr.png logpos.png reflVar.png errorRelCov.png trace.png atmVar.png reflMean.png

filename = 'A9.tgz'

remoteDir = "kmleung@hypersonic.mit.edu:/master/home/kmleung/JPLproject/results/MCMC/A9_initisofit_rank100_constrained/"
localDir = "/Users/KelvinLeung/Documents/JPLproject/results/MCMC/"

subprocess.call(["rsync", "-chavzP", "--stats", "--delete", remoteDir + filename, localDir])








