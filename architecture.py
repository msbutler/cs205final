import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy as sp

from PIL import Image
import os

"""
Defines CNN architecture
"""
def toy_model(x,show_dim = False):

    """
    x: input tensor
    in_channels: number of input channels
    out_channels: number ouf output channels
    show_dim: if true, prints the dimensions of the output tensor
    stride: stride of kernel
    kernel_h: kernel height
    kernel_w: kernel width
    """
    def conv_block(x, in_channels, out_channels, kernel_size, stride, show_dim = False):
        shape = [kernel_size, kernel_size, in_channels, out_channels]
        x = conv_layer(x, shape, stride)
        x = relu(x)
        x = batch_norm(x)
        x = max_pool(x, k=2)
        if show_dim:
            print(x.shape)
        return x

    x = conv_block(x,3,16,25,10,show_dim)

    x = conv_block(x,16,16,5,2,show_dim)

    x = conv_block(x,16,32,5,2,show_dim)

    # flatten output and put through a fully connected layer
    flat1, size1 = flatten_layer(x)
    print(flat1.shape)
    fc1 = fully_connected_layer(flat1, [size1, 64])
    fc1 = relu(fc1)
    if show_dim:
        print(fc1.shape)

    fc2 = fully_connected_layer(fc1, [64, 1])
    if show_dim:
        print(fc2.shape)

    return fc2

# convolutional layer
def conv_layer(x, shape,stride):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    bias = tf.Variable(tf.constant(0.05, shape=[shape[-1]]))

    out = tf.nn.conv2d(input=x, filters=weights, strides=stride, padding='VALID')
    out += bias
    return out

# pooling layer
def max_pool(x, k=2):

    out = tf.nn.max_pool(value=x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')
    return out

# fully connected layer
def fully_connected_layer(x, shape):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    bias = tf.Variable(tf.constant(0.05, shape=[shape[1]]))

    out = tf.matmul(a=x, b=weights)
    out += bias
    return out

# flatten layer
def flatten_layer(x):

    size = x.get_shape()[1:4].num_elements()
    out = tf.reshape(x, [-1,size])
    return out, size

# relu
relu = lambda x: tf.nn.relu(features=x)

# softmax
softmax = lambda x: tf.nn.softmax(logits=x)

# sigmoid
sigmoid = lambda x: tf.nn.sigmoid(x)

# batch norm
batch_norm = lambda x: tf.layers.batch_normalization(x)
