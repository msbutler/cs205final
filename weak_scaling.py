import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from PIL import Image
import os

from utils import *
import semisupervised
import supervised
import time
import sys

def weak(semi = False):
    print ("weak scaling test")

    # gather data
    flooded_img, nonflooded_img, unlabeled_img = prep_data(semi)

    times = []
    data_fracs = [0.25,0.5,0.75,1]
    # data_fracs = [0.1,0.2,0.3,0.5]
    for data_frac in data_fracs:
        print(f"frac of data {data_frac}")
        input_dict = {}
        flooded_img_n = flooded_img[:int(len(flooded_img)*data_frac)]
        nonflooded_img_n = nonflooded_img[:int(len(nonflooded_img)*data_frac)]
        unlabeled_img_n = unlabeled_img[:int(len(unlabeled_img)*data_frac)]
        data_img_n = np.vstack((np.array(flooded_img_n), np.array(nonflooded_img_n))) / 255.
        input_dict["idxs"] = train_test_split(flooded_img_n,nonflooded_img_n,unlabeled_img_n,n=0)

        input_dict["data_img"] = data_img_n
        input_dict["unlabeled_img"] = np.array(unlabeled_img_n)
        begin = time.time()

        if semi==True:
            semisupervised.fit(input_dict,training_iters=iters)

        else:
            supervised.fit(input_dict,training_iters=iters)
        times.append((time.time()-begin)/iters)

    print("Weak Scaling Times: {}".format(times))

    fig,ax = plt.subplots(1,1)
    ax.plot(data_fracs,times)
    ax.set_xlabel('Fraction of Dataset')
    ax.set_ylabel('Ave Time per Epoch (s)')
    fig.savefig("Figures/avg_epoch_seconds_over_datasize.png")

if __name__ == '__main__':
    weak(sys.argv[1])
