import tensorflow as tf
import pandas as pd
import numpy as np
import random
from scipy import ndimage
from matplotlib import pyplot as plt

def dense_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.random.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='W')
        b = tf.Variable(tf.zeros([n_neurons]), name='b')
        Z = tf.matmul(X,W) + b
        print('Shape of {} output: '.format(name), Z.shape)
        if activation is not None:
            return activation(Z)
        else:
            return Z

def conv2d_layer(X, num_filters, filter_size, input_channels, name, activation=None):
    with tf.name_scope(name):
        init = tf.random.truncated_normal((filter_size,filter_size,input_channels,num_filters), stddev=0.1)
        W = tf.Variable(init, name='W')
        b = tf.Variable(tf.zeros((num_filters)))
        Z = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME') + b
        print('Shape of {} output: '.format(name), Z.shape)
        if activation is not None:
            return activation(Z)
        else:
            return Z

def max_pool_layer(X,window_size,name):
    with tf.name_scope(name):
        Z = tf.nn.max_pool(X, ksize=[1,window_size,window_size,1], strides=[1,2,2,1],padding='SAME')
        print('Shape of {} output: '.format(name), Z.shape)
        return Z

def flatten_layer(X,name):
    with tf.name_scope(name):
        Z = tf.reshape(X,[-1, X.shape[1]*X.shape[2]*X.shape[3]])
        print('Shape of {} output: '.format(name), Z.shape)
        return Z

def batch_normalize(Z):
    A = tf.Variable(1,name='A',dtype=tf.float32) # Trainable scale parameter
    B = tf.Variable(0,name='B',dtype=tf.float32) # Trainable offset parameter
    mu = tf.reduce_mean(Z)
    std = tf.math.reduce_std(Z)
    Z_norm = (Z - mu)/std
    Z_bn = (Z_norm * A) + B
    print('Applying batch normalization')
    return Z_bn

def dropout(Z,rate):
    scale = 1/(1-rate) # Used to ensure expectation of sum of layer output remains the same
    shape = tf.shape(Z)
    temp = tf.random.uniform(shape, dtype=Z.dtype)
    mask = tf.cast((temp >= rate),Z.dtype) # Random mask of 1s and 0s
    Z_DO = tf.cond(rate>0, lambda: Z * mask * scale, lambda: Z) # For efficiency, if rate=0 simply return Z
    print('Applying dropout')
    return Z_DO

def plot_digit(digit_values):
    plt.imshow(digit_values.astype(np.float32), cmap="Greys", interpolation='nearest')
    plt.show()

def load_train_data_from_csv():
    train = pd.read_csv('./train.csv')
    train_target_all = train['label']
    train_data_all = train.drop(columns='label')
    train_target_all = train_target_all.values.astype(np.int8)
    train_data_all = train_data_all.values.astype(np.float32)
    example_num = train_data_all.shape[0]
    train_data_all = train_data_all.reshape((example_num,28,28))
    train_data_all = train_data_all[:,:,:,np.newaxis]
    np.save('train_target.npy',train_target_all)
    np.save('train_data.npy',train_data_all)

def load_test_data_from_csv():
    test = pd.read_csv('./test.csv')
    test = test.values.astype(np.float32)
    example_num = test.shape[0]
    test = test.reshape((example_num,28,28))
    test = test[:,:,:,np.newaxis]
    np.save('test_kaggle.npy',test)
    return test

def data_augmentation(X_batch):
    batch_size = X_batch.shape[0]
    rot_angle_deg = random.uniform(-15,15) # random rotations (degrees)
    shift = random.randint(-1,1) # random shifts (pixels)
    image_stack = np.copy(X_batch)
    image_stack_rot = ndimage.rotate(image_stack,rot_angle_deg,axes=(1,2),reshape=False,mode='reflect',order=3)
    image_stack_rot_shift = ndimage.shift(image_stack_rot,(0,shift,shift,0),mode='reflect',order=0)
    X_batch_aug = image_stack_rot_shift
    return X_batch_aug

def next_batch(X,y,size,counter):
    X_batch = X[size*counter:size*(counter+1),:]
    y_batch = y[size*counter:size*(counter+1)]
    X_batch_aug = data_augmentation(X_batch)
    return X_batch_aug, y_batch

def plot_incorrect_digits(val_data,val_target,preds):
    num_per_row = 4
    num_per_col = 2
    for j in range(num_per_col):
        for i in range(num_per_row):
            index = (i + (j*num_per_row))
            digit = val_data[index,:,:,0]
            target_digit = val_target[index]
            pred = preds[index]
            plt.subplot(num_per_col,num_per_row,index+1)
            plt.imshow(digit.astype(np.float32), cmap="Greys", interpolation='nearest')
            plt.title('Target: {}\nPrediction: {}'.format(target_digit,pred))
            plt.axis('off')
    plt.savefig('resources/misclassified_digits.png')
