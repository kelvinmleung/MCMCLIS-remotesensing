import subprocess

filename = 'C1_initMAP_rank175.tgz'

remoteDir = "kmleung@hypersonic.mit.edu:/master/home/kmleung/JPLproject/results/MCMC/"
localDir = "/Users/KelvinLeung/Documents/JPLproject/results/MCMC/"

subprocess.call(["rsync", "-chavzP", "--stats", "--delete", remoteDir + filename, localDir])


'''
./dropbox_uploader.sh upload ../JPLproject/results/MCMC/B5 /
./dropbox_uploader.sh upload ../JPLproject/results/MCMC/B6 /
./dropbox_uploader.sh upload ../JPLproject/results/MCMC/C6 /


'''










