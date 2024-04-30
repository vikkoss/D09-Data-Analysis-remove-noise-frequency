import os
import pandas as pd
import scipy.io as sio

class Data:
    def __init__(self, subject, condition):
        mat = sio.loadmat(f"./Data/ae2224I_measurement_data_subj{subject}_C{condition}.mat")
        #print(mat)
        self.u = mat['u']
        self.t = mat['t']
        self.x = mat['x']
        self.ft = mat['ft']
        self.fd = mat['fd']
        self.e = mat['e']
        self.Hpxd_FC = mat['Hpxd_FC']
        self.Hpe_FC = mat['Hpe_FC']
        self.w_FC = mat['w_FC']




