import subprocess

filename = 'C1_initMAP_rank175.tgz'

remoteDir = "kmleung@hypersonic.mit.edu:/master/home/kmleung/JPLproject/results/MCMC/"
localDir = "/Users/KelvinLeung/Documents/JPLproject/results/MCMC/"

subprocess.call(["rsync", "-chavzP", "--stats", "--delete", remoteDir + filename, localDir])








