import subprocess
subprocess.call(["rsync", "-chavzP", "--stats", "--delete", "kmleung@hypersonic.mit.edu:/master/home/kmleung/JPLproject/results/MCMC", "/Users/KelvinLeung/Documents/JPLproject/results/MCMC/fromCluster"])

indSet = [30,40,90,100,150,160,250,260,425,426]




