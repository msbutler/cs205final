import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from skimage import transform
import random
from PIL import Image
import os
import matplotlib.pyplot as plt

from architecture import *
from config import *

training_iters = 500
batch_size = 4
learning_rate = 1e-3


"""
fits supervised model with input data,
- input_dict: training and test indices given by input dict
- training_iters: number of epochs to train for
"""
def fit(input_dict, training_iters=training_iters):
    data_img = input_dict["data_img"]
    train_idx, test_idx, train_labels, test_labels, dummy1, dummy2 = input_dict["idxs"]


    # define inputs
    x = tf.placeholder(tf.float32, [None, v_dim, h_dim, 3])
    y = tf.placeholder(tf.float32, [None, 1])
    y_train_true = np.array(train_labels).reshape(-1,1)
    y_test_true = np.array(test_labels).reshape(-1,1)

    # run model with placeholder tensors
    pred = toy_model(x,show_dim = True)

    # sharpen
    # pred = sharpen(pred)

    # define loss
    cross_entropy = tf.losses.sigmoid_cross_entropy(logits=pred, multi_class_labels=y)
    cost = tf.reduce_mean(cross_entropy)

    # define accuracy
    pred_class = tf.round(sigmoid(pred))
    pred_correct = tf.equal(pred_class, tf.cast(y, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(pred_correct, tf.float32))

    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # initialize variables
    init = tf.global_variables_initializer()

    # config = tf.ConfigProto(device_count = {'GPU': num_gpu})

    # train model
    with tf.Session() as sess:
        devices = sess.list_devices()
        for d in devices:
            print(d)

        sess.run(init)

        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        best_acc = 0.
        summary_writer = tf.summary.FileWriter('./Output', sess.graph)
        num_batches = len(train_idx)//batch_size

        for i in range(training_iters):

            # Reset metrics
            loss_total = 0
            acc_total = 0
            train_results = []

            # Run optimization
            # Calculate batch loss and accuracy
            for batch in range(num_batches):
                batch_x = data_img[train_idx,:,:,:][batch*batch_size:min((batch+1)*batch_size,len(train_idx))]
                batch_y = y_train_true[batch*batch_size:min((batch+1)*batch_size,len(y_train_true))]

                feed_dict={x: batch_x, y: batch_y}
                opt = sess.run(optimizer, feed_dict=feed_dict)
                loss, acc, pred_labels = sess.run([cost, accuracy, pred_class], feed_dict=feed_dict)
                loss_total += loss
                acc_total += acc
                train_results.append(pred_labels)

            # Average metrics
            ave_loss = loss_total/num_batches
            ave_acc = acc_total/num_batches

            # Calculate accuracy for all test images
            valid_loss, test_acc, test_results = sess.run([cost, accuracy, pred_class],
                                    feed_dict={x: data_img[test_idx,:,:,:], y : y_test_true})

            # Update metrics
            train_loss.append(ave_loss)
            test_loss.append(valid_loss)
            train_accuracy.append(ave_acc)
            test_accuracy.append(test_acc)
            if test_acc > best_acc:
                best_model_train_labels = tf.stack(tf.reshape(tf.stack(train_results),[-1,1])).eval()
                best_model_test_labels = test_results
                best_acc = test_acc

            # Print metrics
            print("Iter " + str(i) + ", Loss= " + \
                        "{:.6f}".format(ave_loss) + ", Training Accuracy= " + \
                        "{:.5f}".format(ave_acc)+ \
                        " ,Testing Accuracy:","{:.5f}".format(test_acc))
        summary_writer.close()

    fig,ax=plt.subplots(1,1)
    ax.plot(train_loss)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    fig.savefig("Figures/Train Loss - Balanced Supervised.png")

    fig,ax=plt.subplots(1,1)
    ax.plot(train_accuracy)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    fig.savefig("Figures/Train Accuracy - Balanced Supervised.png")

    fig,ax=plt.subplots(1,1)
    ax.plot(test_accuracy)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    fig.savefig("Figures/Test Accuracy - Balanced Supervised.png")
