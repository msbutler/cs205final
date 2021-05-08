import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from PIL import Image
import os

from utils import *
import semisupervised



def run(semi = False):

    # gather data
    flooded_img, nonflooded_img, unlabeled_img = prep_data(semi)
    data_img = np.vstack((np.array(flooded_img), np.array(nonflooded_img))) / 255.

    #split data
    idxs = train_test_split(flooded_img,nonflooded_img,unlabeled_img,n)
    label_train_idx, label_test_idx, train_labels, test_labels, unlabel_train_idx, unlabel_test_idx = idxs

    if semi==True:
        semisupervised.fit()
    print("Loaded!")



if __name__ == '__main__':
    run(True)

    

    