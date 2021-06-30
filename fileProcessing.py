import numpy as np
import json
from scipy.io import loadmat
from fsplit.filesplit import Filesplit


class FileProcessing:


    def __init__(self):
        print('\n')

    def loadWavelength(self, wvFile): #'setup/data/wavelengths.txt'
        wvl, wv, wvr = np.loadtxt(wvFile).T
        # mat = loadmat(wvFile)
        # self.wv = mat['wl'][0]
        self.wv = wv * 1000

    def loadReflectance(self, refFile):
        data = np.loadtxt(refFile).T
        wvRaw = data[0]
        refRaw = data[1]
        self.ref = np.interp(self.wv, wvRaw, refRaw)

    def loadRadiance(self, datamatfile):
        mat = loadmat(datamatfile)
        self.radiance = mat['meas'][0]

    def loadConfig(self, configFile):
        with open(configFile, 'r') as f:
            self.config = json.load(f)
    
    def getFiles(self):
        return self.wv, self.ref, self.radiance, self.config

    def splitFile(self, filename, output):
        fs = Filesplit()
        def split_cb(f, s):
            print("file: {0}, size: {1}".format(f, s))
        fs.split(file=filename, split_size=9000000000, output_dir=output, callback=split_cb)

    def mergeFile(self, inputdir):
        fs = Filesplit()
        def merge_cb(f, s):
            print("file: {0}, size: {1}".format(f, s))
        fs.merge(input_dir=inputdir, callback=merge_cb)

    def thinMCMCFile(self, inputdir, thinning):
        x_vals =  np.load(inputdir + 'MCMC_x.npy', mmap_mode='r')
        x_vals_thin = x_vals[:,::thinning]
        np.save(inputdir + 'MCMC_x_thin.npy', x_vals_thin)
        


        