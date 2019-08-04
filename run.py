import tensorflow as tf
import pandas as pd
import numpy as np
from math import floor
from datetime import datetime
from time import time
import os
import sys
import random
from tqdm import tqdm
from tools import *

########################################################
# Variable parameters here:
img_dim = 28
max_n_epochs = 50
patience = 7 # epochs to continue beyond last highest validation score
batch_size = 64
dropout_rate = 0.4
percent_data_for_training = 90

TRAIN_MODEL = True
TEST_MODEL = True
SHOW_INCORRECT = False

root_logdir = 'tf_logs'
root_modeldir = './models'
########################################################

t1 = time()
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = '{}/run_{}/'.format(root_logdir, now)
if TRAIN_MODEL:
    modeldir = '{}/run_{}/'.format(root_modeldir, now)
else:
    modeldir = sys.argv[1]

if TRAIN_MODEL:
    if not os.path.exists(logdir):
        os.makedirs(logdir)

if not os.path.exists(modeldir):
    os.makedirs(modeldir)

load_train_data_from_csv()
train_target_all = np.load('train_target.npy')
train_data_all = np.load('train_data.npy')

mean_train = train_data_all.mean()
std_train = train_data_all.std()
train_data_all = (train_data_all - mean_train)/std_train

shuffle_index = np.random.permutation(train_data_all.shape[0])
train_data_all, train_target_all = train_data_all[shuffle_index], train_target_all[shuffle_index]
train_index = floor(train_data_all.shape[0]*(percent_data_for_training/100))
train_data, train_target = train_data_all[:train_index], train_target_all[:train_index]
val_data, val_target = train_data_all[train_index:], train_target_all[train_index:]

rate = tf.placeholder(tf.float32, shape=())
zero_dropout = 0

X = tf.placeholder(tf.float32, shape=(None, img_dim, img_dim, 1), name = 'X')
y = tf.placeholder(tf.int32, shape=(None), name = 'y')

with tf.name_scope('NeuralNetwork'):
    print('*'*20)
    print('Building graph:')
    Z1 = conv2d_layer(X, num_filters=32, filter_size=3, input_channels=1, name='conv1',activation=tf.nn.relu)
    Z2 = batch_normalize(Z1)
    Z3 = conv2d_layer(Z2, num_filters=32, filter_size=3, input_channels=32, name='conv2',activation=tf.nn.relu)
    Z4 = batch_normalize(Z3)
    Z5 = max_pool_layer(Z4,window_size=2,name='maxpool1')
    Z6 = dropout(Z5,rate=rate)
    Z7 = conv2d_layer(Z6, num_filters=64, filter_size=3, input_channels=32, name='conv3',activation=tf.nn.relu)
    Z8 = batch_normalize(Z7)
    Z9 = conv2d_layer(Z8, num_filters=64, filter_size=3, input_channels=64, name='conv4',activation=tf.nn.relu)
    Z10 = batch_normalize(Z9)
    Z11 = max_pool_layer(Z10,window_size=2,name='maxpool2')
    Z12 = dropout(Z11,rate=rate)
    Z13 = flatten_layer(Z12,name='flatten1')
    Z14 = dense_layer(Z13,n_neurons=128,name='dense1',activation=tf.nn.relu)
    Z15 = batch_normalize(Z14)
    Z16 = dropout(Z15,rate=rate)
    logits = dense_layer(Z16,n_neurons=10,name='logits')
    print('*'*20)
    print()

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

with tf.name_scope('GradDecent'):
    optimiser = tf.train.AdamOptimizer()
    training_step = optimiser.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    predictions_prob = tf.nn.softmax(logits)

initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

acc_train_summary = tf.summary.scalar('Acc_train', accuracy)
acc_val_summary = tf.summary.scalar('Acc_val', accuracy)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:
    initializer.run()
    highest_acc_val = 0
    steps_since_highest = 0
    if TRAIN_MODEL:
        for epoch in range(max_n_epochs):
            a = time()
            for counter in tqdm(range(len(train_target) // batch_size)):
                X_batch, y_batch = next_batch(train_data,train_target,batch_size,counter)
                sess.run(training_step, feed_dict={X: X_batch, y: y_batch, rate: dropout_rate})
            acc_train = accuracy.eval(feed_dict={X: train_data, y: train_target, rate: zero_dropout})
            acc_val = accuracy.eval(feed_dict={X: val_data, y: val_target, rate: zero_dropout})
            if acc_val > highest_acc_val:
                highest_acc_val = acc_val
                steps_since_highest = 0
                saver.save(sess,modeldir+'model.ckpt')
            else:
                steps_since_highest += 1
                print('Steps since highest acc: ',steps_since_highest)
            b = time()
            # print('Epoch time: ', '{:.3f}'.format(b-a), ' secs')
            acc_train_summary_str = acc_train_summary.eval(feed_dict={X: train_data, y: train_target, rate: zero_dropout})
            acc_val_summary_str = acc_val_summary.eval(feed_dict={X: val_data, y: val_target, rate: zero_dropout})
            file_writer.add_summary(acc_train_summary_str,epoch)
            file_writer.add_summary(acc_val_summary_str,epoch)
            print(epoch+1, "Train acc: ", acc_train, "Val acc: ", acc_val)
            if steps_since_highest >= patience:
                break

    if TEST_MODEL:
        saver.restore(sess,modeldir+'model.ckpt')
        load_test_data_from_csv()
        test_data = np.load('test_kaggle.npy')
        test_data = (test_data - mean_train)/std_train
        Z = logits.eval(feed_dict={X: test_data, rate: zero_dropout})
        y_pred = np.argmax(Z, axis=1)
        pred_csv = pd.DataFrame(y_pred, columns= ['Label'])
        pred_csv.index += 1
        pred_csv.to_csv('submission.csv', index_label='ImageId' )

    if SHOW_INCORRECT:
        saver.restore(sess,modeldir+'model.ckpt')
        correct_preds = correct.eval(feed_dict={X: val_data, y: val_target, rate: zero_dropout})
        preds_prob = predictions_prob.eval(feed_dict={X: val_data, y: val_target, rate: zero_dropout})
        preds_hard = np.argmax(preds_prob, axis=1)
        incorrect_digits = val_data[~correct_preds]
        incorrect_targets = val_target[~correct_preds]
        incorrect_preds = preds_hard[~correct_preds]
        plot_incorrect_digits(incorrect_digits,incorrect_targets,incorrect_preds)

file_writer.close()
t2 = time()
t_tot = t2-t1
print('Total time elapsed: ', '{:.2f}'.format(t_tot), ' secs')
