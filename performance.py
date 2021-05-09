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

# set to max number of gpus on the instance
max_gpus = 1
iters = 1

def run(semi = False):
    input_dict = {}
    # gather data
    flooded_img, nonflooded_img, unlabeled_img = prep_data(semi)
    data_img = np.vstack((np.array(flooded_img), np.array(nonflooded_img))) / 255.

    input_dict["data_img"] = data_img
    input_dict["unlabeled_img"] = unlabeled_img

    #split data
    input_dict["idxs"] = train_test_split(flooded_img,nonflooded_img,unlabeled_img,n)

    times = []

    print(f"FOR {max_gpus} GPUS")
    for i in range(max_gpus+1):
        begin = time.time()
        print("Test on {i} GPUS!")
        gpus = ','.join(list(range(i)))
        print(gpus)
        os.environ["CUDA_VISIBLE_DEVICES"]=gpus

        if semi==True:
            semisupervised.fit(input_dict,training_iters=iters)
        
        else:
            supervised.fit(input_dict,training_iters=iters)
        times.append((time.time()-begin)/iters)

    plt.plot(times,range(max_gpus))
    plt.savefig("avg_epoch_time_over_gpu.png")

if __name__ == '__main__':
    run(False)