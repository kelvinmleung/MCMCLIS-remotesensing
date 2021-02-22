import subprocess

filename = 'C1_initMAP_rank175.tgz'

remoteDir = "kmleung@hypersonic.mit.edu:/master/home/kmleung/JPLproject/results/MCMC/"
localDir = "/Users/KelvinLeung/Documents/JPLproject/results/MCMC/"

subprocess.call(["rsync", "-chavzP", "--stats", "--delete", remoteDir + filename, localDir])


'''
./dropbox_uploader.sh upload ../JPLproject/results/MCMC/B7 /
./dropbox_uploader.sh upload ../JPLproject/results/MCMC/B8 /
./dropbox_uploader.sh upload ../JPLproject/results/MCMC/C7 /
./dropbox_uploader.sh upload ../JPLproject/results/MCMC/C8 /


'''


## scrap code for zipping plots

# zip the plots
# listFiles = []
# listFiles = listFiles + ['reflMean.png', 'reflError.png', 'atmMean.png', 'atmError.png','reflVar.png', 'atmVar.png', 'errorRelCov.png', 'trace.png', 'autocorr.png', 'logpos.png', 'acceptance.png', 'runtime.txt']
# numPairs = int(len(indSet) / 2)
# for i in range(numPairs):
#     listFiles = listFiles + ['2D_' + str(indSet[2*i]) + '-' + str(indSet[2*i+1]) + '.png']
# subprocess.call(['tar', '--directory', setup.mcmcDir, '-zcvf', '../results/MCMC/' + mcmcfolder + '.tgz'] + listFiles)









