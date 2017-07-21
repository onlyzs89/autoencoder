import numpy as np
import tensorflow as tf
import random
import pickle
import cv2
from scipy import stats
from sklearn import metrics

import reader2

FILE_NAME = 'prototype_1/data_batch_'
WIDTH = 32
HEIGHT = 32
DEPTH = 3
LOOP = 10000
BATCH_SIZE = 100

# TF
def weight_variable_conv(shape):
    kW = shape[0]
    kH = shape[1]
    outPlane = shape[3]
    n = kW*kH*outPlane
    stddev = tf.sqrt(2.0/n)
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable_conv(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def de_conv2d(x, W,output_shape,strides):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=strides, padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
def encoder(x):
    W_conv1 = weight_variable([3, 3, DEPTH, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([3, 3, 32, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([3, 3, 16, 8])
    b_conv3 = bias_variable([8])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    
    return h_pool3
    
def decoder(x):
    de_W_conv3 = weight_variable([3, 3, 16, 8])
    de_b_conv3 = bias_variable([16])
    de_h_conv3 = tf.nn.relu(de_conv2d(x, de_W_conv3, [BATCH_SIZE, WIDTH//4, HEIGHT//4, 16], [1,2,2,1]) + de_b_conv3)
    
    de_W_conv2 = weight_variable([3, 3, 32, 16])
    de_b_conv2 = bias_variable([32])
    de_h_conv2 = tf.nn.relu(de_conv2d(de_h_conv3, de_W_conv2, [BATCH_SIZE, WIDTH//2, HEIGHT//2, 32], [1,2,2,1]) + de_b_conv2)
    
    de_W_conv1 = weight_variable([3, 3, DEPTH, 32])
    de_b_conv1 = bias_variable([DEPTH])
    de_h_conv1 = tf.nn.relu(de_conv2d(de_h_conv2, de_W_conv1, [BATCH_SIZE, WIDTH, HEIGHT, DEPTH], [1,2,2,1]) + de_b_conv1)
    
    return de_h_conv1

def training():
    image_list = []
    for i in range(1,4):
        # read train file
        r = reader2.fileReader(FILE_NAME + str(i))
        label_list_sub, image_list_sub = r.read_file(False)
        image_list.extend(image_list_sub)
    
    #shuffle
    train_image = []
    index_shuf = list(range(len(image_list)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        train_image.append(image_list[i])
    
    ckpt = tf.train.get_checkpoint_state('./')
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        saver.restore(sess, last_model)
    
    else:
        sess.run(tf.global_variables_initializer())
        
        for i in range(LOOP):
            offset = random.randint(0, int(len(train_image)//BATCH_SIZE)-1) * BATCH_SIZE
            train_image_batch = train_image[offset:(offset + BATCH_SIZE)]
            optimizer.run(feed_dict={x: train_image_batch})
            if i%100==0:
                cost_eval = cost.eval(feed_dict={x: train_image_batch})
                print('cost: ' + str(cost_eval))
                saver.save(sess, "./model.ckpt")
                
                if cost_eval<0.003:
                    break
    
    offset = random.randint(0, int(len(train_image)//BATCH_SIZE)-1) * BATCH_SIZE
    train_image_batch = train_image[offset:(offset + BATCH_SIZE)]
    diff_eval = diff.eval(feed_dict={x: train_image_batch[:BATCH_SIZE]})
    mean_val = np.mean(diff_eval)
    sem_val  = stats.sem(diff_eval)
    ci = stats.t.interval(0.95, len(diff_eval)-1, loc=mean_val, scale=sem_val)
    print("lossの95%%信頼区間： %(ci)s" %locals())
    
    return ci[1]


def testing(thres):
    ckpt = tf.train.get_checkpoint_state('./')
    last_model = ckpt.model_checkpoint_path
    saver.restore(sess, last_model)

    image_list = []
    label_list = []
    for i in range(4,6):
        r = reader2.fileReader(FILE_NAME + str(i))
        label_list_sub, image_list_sub = r.read_file(False)
        image_list.extend(image_list_sub)
        label_list.extend(label_list_sub)
    image_list = image_list[:50]
    label_list = label_list[:50]
    for i in range(1,6):
        label_list_sub, image_list_sub = r.read_file(True)
        image_list.extend(image_list_sub)
        label_list.extend(label_list_sub)
    image_list = image_list[:100]
    label_list = label_list[:100]
    
    pred_list = []
    res = diff.eval(feed_dict={x:image_list})
    for i in res:
        if i>thres*1.2:
            pred_list.append(1)
        else:
            pred_list.append(0)
    
    print(metrics.classification_report(label_list, pred_list))
    print(metrics.confusion_matrix(label_list, pred_list))

    
# tf model
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, WIDTH*HEIGHT*DEPTH], name='input')
x_true = tf.reshape(x, [-1,WIDTH,HEIGHT,DEPTH])
encoded = encoder(x_true)
x_pred= decoder(encoded)

diff = tf.reduce_mean(tf.pow(x_true - x_pred, 2),(1,2,3))
cost = tf.reduce_mean(diff)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

# create saver
saver = tf.train.Saver()

ci = training()
testing(ci)
