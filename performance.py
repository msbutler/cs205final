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

def strong(semi = False):
    print("strong scaling test")
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
        print(f"Test on {i} GPUS!")
        l = [str(x) for x in list(range(i))]
        gpus = ','.join(l)
        print(gpus)
        os.environ["CUDA_VISIBLE_DEVICES"]=gpus

        with tf.Session() as sess:
            devices = sess.list_devices()
            for d in devices:
                print(d)

        if semi==True:
            semisupervised.fit(input_dict,training_iters=iters)
        
        else:
            supervised.fit(input_dict,training_iters=iters)
        times.append((time.time()-begin)/iters)

    plt.plot(range(max_gpus),times)
    plt.savefig("avg_epoch_seconds_over_gpu.png")

def weak(semi = False):
    print ("weak scaling test")

    # gather data
    flooded_img, nonflooded_img, unlabeled_img = prep_data(semi)

    times = []
    data_fracs = [0.25,0.5,0.75,1]
    for data_frac in data_fracs:
        print(f"frac of data {data_frac}")

        flooded_img_n = flooded_img[:int(len(flooded_img)*data_frac)]
        non_flooded_img_n = nonflooded_img[:int(len(nonflooded_img)*data_frac)]
        unlabeled_img_n = unlabeled_img[:int(len(unlabeled_img)*data_frac)]
        data_img = np.vstack((np.array(flooded_img_n), np.array(nonflooded_img_n))) / 255.

        input_dict["data_img"] = data_img
        input_dict["unlabeled_img"] = unlabeled_img_n
        begin = time.time()

        if semi==True:
            semisupervised.fit(input_dict,training_iters=iters)
        
        else:
            supervised.fit(input_dict,training_iters=iters)
        times.append((time.time()-begin)/iters)

    plt.plot(data_fracs,times)
    plt.savefig("avg_epoch_seconds_over_datasize.png")
if __name__ == '__main__':
    weak(False)
    strong(False)