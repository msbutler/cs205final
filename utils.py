
from PIL import Image
import os
import random
import numpy as np
from config import h_dim, v_dim, n
import skimage as sk
from skimage import transform

""" compresses all .jpg files in the directory and subdirectory to have a certain basewidth"""
def compress_im(root,basewidth):
    print(f"compress imgs in {root}")
    
    for ob in os.listdir(root):
        file_path = os.path.join(root,ob)
        if os.path.isdir(file_path):
            compress_im(file_path,basewidth)
        elif ob[-3:] == "jpg":
            img = Image.open(file_path)
            comp_frac = basewidth/img.size[0]
            assert (comp_frac <=1.)
            hsize = int(img.size[1] * comp_frac)
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)
            img.save(file_path)
            
            
def prep_data(semi = False):
    print("Loading Data from Memory")

    root = "Train"
    label_flood_dir = os.path.join(root,'Labeled','Flooded','image')
    label_nonflood_dir = os.path.join(root,'Labeled','Non-Flooded','image')
    unlabel_dir = os.path.join(root,'Unlabeled/image/')

    flooded_img = []
    nonflooded_img = []
    unlabeled_img = []

    for file in os.listdir(label_flood_dir):
        image = Image.open(os.path.join(label_flood_dir, file))
        image = np.array(image.resize((h_dim,v_dim)))
        if semi == True:
            flooded_img.append(rotate_img(image))
        else:
            flooded_img.append(image)
        
    for file in os.listdir(label_nonflood_dir):
        image = Image.open(os.path.join(label_nonflood_dir, file))
        image = np.array(image.resize((h_dim,v_dim)))
        if semi == True:
            nonflooded_img.append(rotate_img(image))
        else:
            nonflooded_img.append(image)

    print("Flooded Image Shape: {}".format(flooded_img[0].shape))
    print("Non_Flooded Image Shape: {}".format(nonflooded_img[0].shape))

    if semi == True:
        for file in os.listdir(unlabel_dir):
            image = Image.open(os.path.join(unlabel_dir, file))
            image = np.array(image.resize((h_dim,v_dim)))
            unlabeled_img.append(rotate_img(image))
            unlabeled_img.append(rotate_img(image))
        print("Unlabeled Image Shape: {}".format(unlabeled_img[0].shape))

    return flooded_img, nonflooded_img, unlabeled_img


""" splits labeled flooded data, labeled unflooded, and unlabelled data into training and test data

if n = 0, then all data will be used; else, n samples from each data set will be selected for train-test split. 
So if n>0, n must be min(len(flooded_img),len(non_flooded_img),len(unlabelled_img))

assumes labelled data live in the data_img and that flooded appears before non-flooded 

flooded_img: array of flooded images
nonflooded_img: array of unflooded images
unlabelled_img: array of unflooded images

return

train_idx: indices for data_img for training data
test_idx: indices for data_img for test data
train_labels: the labels associated with the training data
test_labels: the labels associated with test data

*optional return values if nonzero unlablled img is passed in*
unlabel_train_idx: the indices for unlabelled_img for unlabelled training data
unlabel_test_idx: the indices for unlabelled_img for unlabelled test data
"""
def train_test_split(flooded_img, nonflooded_img, unlabelled_img=np.array([]),n=0):
    
    train_idx = []
    test_idx = []
    
    #index range to select from (flooded img range, non flooded image range)
    for s_i,e_i in [(0,len(flooded_img)),(len(flooded_img),len(flooded_img)+len(nonflooded_img))]:
        
        if n == 0:
            n_s = e_i-s_i
        else:
            assert n <= e_i-s_i
            n_s = n
            
        s = int(np.floor(0.8*n_s)) # number of images for training
        #get all poss indexes for set
        s_idx = list(range(s_i,e_i))
        random.shuffle(s_idx)
        
        train_idx.extend(s_idx[:s])      #first s images are for training
        test_idx.extend(s_idx[s:n_s])      # next n-s images are for testing. 
        print(f"Data len {e_i-s_i}; n is {n_s} ;train size {s}, test size {n_s-s}")
    
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    train_idx.sort()
    test_idx.sort()
    
    # data live in data_img, and flooded data appears first. 1 = flooded
    train_labels = [1 if x<len(flooded_img) else 0 for x in train_idx]
    test_labels = [1 if x<len(flooded_img) else 0 for x in test_idx]

    print(f"Training Indices len {len(train_idx)}")
    print(f"Testing Indices len {len(test_idx)}")
    
    if len(unlabelled_img)== 0:
        print("\t dummy unlabel idx")
        unlabel_train_idx = np.array([])
        unlabel_test_idx = np.array([])

    else:
        if n ==0:
            n_s = len(unlabelled_img)
        else:
            assert n <= len(unlabelled_img)
            n_s = n
            
        u_idx = list(range(n_s))
        #random.shuffle(u_idx)
        unlabel_train_idx = np.array(u_idx[:s])
        unlabel_test_idx = np.array(u_idx[s:n_s])
        #unlabel_train_idx.sort()
        #unlabel_test_idx.sort()
        print(f"Unlabelled Data len {len(unlabelled_img)}; n is {n_s} ;train size {s}, test size {n_s-s}")

    return train_idx, test_idx, train_labels, test_labels, unlabel_train_idx, unlabel_test_idx

def rotate_img(image):
    random_degree = random.uniform(-25, 25) #25% from left or right
    return sk.transform.rotate(image, random_degree)

def noise_img(image):
    return sk.util.random_noise(image)
