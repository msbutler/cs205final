import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import skimage as sk
from skimage import transform
import random
from PIL import Image
import os
import matplotlib.pyplot as plt

from architecture import *
from config import *

training_iters = 200
batch_size = 4
learning_rate = 1e-3

# Google paper section 3.5 says 100 is a good place to start for w_unlabeled
# Google paper also suggests ramping up value to 100 over first 16,000 epochs
w_unlabeled = 100. 

k = 2 # augment images k times

def generate_guess_label(pred_u_raw, k):
    # guess label = average prediction over k augmentations of same image
    # num_images = pred_u_raw.shape[0].value / k # Throws error of NoneType and int since pred_u_raw.shape[0].value is None
    
    try:
        num_images = int(pred_u_raw.shape[0].value / k)

        idx = 0
        temp_labels = []
        for i in range(num_images):
            temp_labels.append(tf.reduce_mean(pred_u_raw[idx:idx+k]))
            idx += k

        # repeat label for each augmentation
        guess_labels = tf.repeat(tf.stack(temp_labels), k)

        # reshape and remove gradient tracking
        guess_labels = tf.reshape(guess_labels, (-1,1))
        guess_labels = tf.stop_gradient(guess_labels)

        return guess_labels

    except TypeError:
      
      return pred_u_raw

def sharpen(p):
    T = 0.5
    pred = p**(1./T)/(p**(1./T) + (1.-p)**(1./T))
    return pred


"""
fits semi-supervised model with input data,
- input_dict: training and test indices given by input dict
- training_iters: number of epochs to train for
"""

def fit(input_dict, training_iters = training_iters):
    data_img = input_dict["data_img"]
    unlabeled_img = input_dict["unlabeled_img"]
    label_train_idx, label_test_idx, train_labels, test_labels, unlabel_train_idx, unlabel_test_idx = input_dict["idxs"]


    # define inputs
    x = tf.placeholder(tf.float32, [None, v_dim, h_dim, 3], 'x') # labeled images (augmented)
    u = tf.placeholder(tf.float32, [None, v_dim, h_dim, 3], 'u') # unlabeled images (augmented)
    y = tf.placeholder(tf.float32, [None, 1], 'y') # labels
    train_labels = np.array(train_labels).reshape(-1,1)
    test_labels = np.array(test_labels).reshape(-1,1)

    # run model with placeholder tensors (feed forward pass)
    pred_x = toy_model(x)
    pred_u_raw = toy_model(u)
    pred_u_raw = sigmoid(pred_u_raw)

    # calculate guess labels for unlabeled images 
    pred_u = generate_guess_label(pred_u_raw, k) # average predictions across same unlabelled images

    # sharpen guess labels 
    pred_u = sharpen(pred_u)

    # define loss
    cross_entropy = tf.losses.sigmoid_cross_entropy(logits=pred_x, multi_class_labels=y)
    labeled_loss = tf.reduce_mean(cross_entropy)
    unlabeled_loss = tf.nn.l2_loss(pred_u - pred_u_raw)
    cost = labeled_loss + w_unlabeled*unlabeled_loss

    # define accuracy
    pred_u_class = tf.round(pred_u_raw)
    pred_x_class = tf.round(sigmoid(pred_x))
    pred_correct = tf.equal(pred_x_class, tf.cast(y, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(pred_correct, tf.float32))

    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # initialize variables
    init = tf.global_variables_initializer()

    # train model
    with tf.Session() as sess:
        sess.run(init)
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        best_acc = 0.
        summary_writer = tf.summary.FileWriter('./Output', sess.graph)
        num_batches = len(label_train_idx)//batch_size
        
        for i in range(training_iters):
            
            # Reset metrics
            loss_total = 0
            acc_total = 0
            train_labeled_results = []
            train_unlabeled_results = []

            # Run optimization 
            # Calculate batch loss and accuracy
            for batch in range(num_batches):
                batch_x = data_img[label_train_idx,:,:,:][batch*batch_size:min((batch+1)*batch_size, len(label_train_idx))]
                batch_u = unlabeled_img[unlabel_train_idx,:,:,:][batch*k*batch_size:min((batch+1)*k*batch_size, len(unlabel_train_idx))]
                batch_y = train_labels[batch*batch_size:min((batch+1)*batch_size, len(train_labels))]

                feed_dict={x: batch_x, u: batch_u, y: batch_y}
                opt = sess.run(optimizer, feed_dict=feed_dict)
                loss, acc, pred_x_labels, pred_u_labels  = sess.run([cost, accuracy, pred_x_class, pred_u_class], 
                                                                    feed_dict=feed_dict)
                loss_total += loss
                acc_total += acc
                train_labeled_results.append(pred_x_labels)
                train_unlabeled_results.append(pred_u_labels)

            # Average metrics
            ave_loss = loss_total/num_batches
            ave_acc = acc_total/num_batches
            

            # Calculate accuracy for all test images
            valid_loss, test_acc, test_labeled_results, test_unlabeled_results = sess.run([cost, accuracy, pred_x_class, pred_u_class],
                                    feed_dict={x: data_img[label_test_idx,:,:,:], u: unlabeled_img[unlabel_test_idx,:,:,:], y : test_labels})
            
            train_loss.append(ave_loss)
            test_loss.append(valid_loss)
            train_accuracy.append(ave_acc)
            test_accuracy.append(test_acc)
            if test_acc > best_acc:
                #best_model_train_labeled = tf.stack(tf.reshape(tf.stack(train_labeled_results),[-1,1])).eval()
                #best_model_train_unlabeled = tf.stack(tf.reshape(tf.stack(train_unlabeled_results),[-1,1])).eval()
                best_model_test_labeled = test_labeled_results
                best_model_test_unlabeled = test_unlabeled_results
                best_acc = test_acc
            
            if i%10 == 0:
                print("Iter " + str(i) + ", Loss= " + \
                    "{:.6f}".format(ave_loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(ave_acc) + ", Testing Accuracy= " + \
                    "{:.5f}".format(test_acc))

        summary_writer.close()
