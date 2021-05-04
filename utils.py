
from PIL import Image
import os
import random
import numpy as np

""" compresses all .jpg files in the directory and subdirectory to have a certain basewidth"""
def compress_im(root,basewidth):
    print(f"compress imgs in {root}")
    
    for ob in os.listdir(root):
        file_path = os.path.join(root,ob)
        if os.path.isdir(file_path):
            compress_im(file_path,basewidth)
        elif ob[-3:] == "jpg":
            #print(file_path)
            #print(os.path.getsize(file_path))
            img = Image.open(file_path)
            img.show()
            comp_frac = basewidth/img.size[0]
            assert (comp_frac <=1.)
            hsize = int(img.size[1] * comp_frac)
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)
            img.show()
            img.save(file_path)
            #print(os.path.getsize(file_path))
            #assert 0 ==1
            
def train_test_split(n, flooded_img, nonflooded_img, unlabelled_img=np.array([])):
    idxs = list(range(n))
    s = int(np.floor(0.8*n)) # number of images for training
    
    train_idx = []
    test_idx = []
    
    #index range to select from (flooded img range, non flooded image range)
    for s_i,e_i in [(0,len(flooded_img)),(len(flooded_img),len(nonflooded_img)+len(flooded_img))]:
        
        #print(s_i,e_i,n,s)
        #get all poss indexes for set
        s_idx = list(range(s_i,e_i))
        random.shuffle(s_idx)
        
        train_idx.extend(s_idx[:s])      #first s images are for training
        test_idx.extend(s_idx[s:n])      # next n-s images are for testing. 
        #print(len(train_idx))
    
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    train_idx.sort()
    test_idx.sort()
    
    train_labels = [1 if x<len(flooded_img) else 0 for x in train_idx]
    test_labels = [1 if x<len(flooded_img) else 0 for x in test_idx]

    print("Training Index: {}".format(train_idx))
    print("Testing Index: {}".format(test_idx))
    if len(unlabelled_img)== 0:
        return train_idx, test_idx, train_labels, test_labels
    else:
        
        u_idx = list(range(n))
        assert n < len(unlabelled_img)
        random.shuffle(u_idx)
        
        unlabel_train_idx = np.array(u_idx[:s])
        unlabel_test_idx = np.array(u_idx[s:n])
        unlabel_train_idx.sort()
        unlabel_test_idx.sort()
        print("Unlabeled Training Index: {}".format(unlabel_train_idx))
        return train_idx, test_idx, train_labels, test_labels, unlabel_train_idx, unlabel_test_idx
