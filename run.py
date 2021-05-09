
import numpy as np
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
    input_dict["unlabeled_img"] = np.array(unlabeled_img)

    #split data
    input_dict["idxs"] = train_test_split(flooded_img,nonflooded_img,unlabeled_img,n=50)

    if semi==True:
        semisupervised.fit(input_dict)
    
    else:
        supervised.fit(input_dict)

    print("Loaded!")



if __name__ == '__main__':
    run(True)

    

    
