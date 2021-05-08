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



def run(semi = False):
    input_dict = {}
    # gather data
    flooded_img, nonflooded_img, unlabeled_img = prep_data(semi)
    data_img = np.vstack((np.array(flooded_img), np.array(nonflooded_img))) / 255.

    input_dict["data_img"] = data_img
    input_dict["unlabeled_img"] = unlabeled_img

    #split data
    input_dict["idxs"] = train_test_split(flooded_img,nonflooded_img,unlabeled_img,n)

     
    for i in range(4):
        print("Test if {i} GPUS!")
        

        if semi==True:
            semisupervised.fit(input_dict,training_iters=1)
        
        else:
            supervised.fit(input_dict,training_iters=1)


if __name__ == '__main__':
    run(False)