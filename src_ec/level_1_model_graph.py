import time
import numpy as np
import tflearn
from sklearn.model_selection import train_test_split
import random
import cPickle
import copy
import math
import sys
sys.path.append('/home/liy0f/ec_project/data_and_feature/')
import protein_sequence_process_functions as p_func
import Pfam_pickle_file_to_array_encoding as Pfam
import tensorflow as tf
import os
from level_2_preprocess import load_level_2_data
from evaluate_model import *

start=time.time()

#Global variable, MAX_LENGTH is the maximum length of all sequence.
TEST=True
DROPOUT=False
MAX_LENGTH=5000
TYPE_OF_AA=20
DOMAIN=16306
LOAD=False
train_ratio=0.8
level=1
n_class=1

batch_size = 30
train_steps = 3000
if TEST==True:
    file_output=False
else:
    file_output=True

#functions to generate variables, like weight and bias
def weight_variable(shape):
    import math
    if len(shape)>2:
        weight_std=math.sqrt(2.0/(shape[0]*shape[1]*shape[2]))
    else:
        weight_std=0.01
    initial=tf.truncated_normal(shape,stddev=weight_std)
    return tf.Variable(initial,name='weights')

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name='bias')

#functions to generate convolutional layer and pooling layer
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def aver_pool2d(x,row,col):
    #Be careful about the dimensionality reduction of pooling and strides setting
    return tf.nn.avg_pool(x,ksize=[1,row,col,1],strides=[1,row,col,1],padding='SAME')
def max_pool2d(x,row,col):
    return tf.nn.max_pool(x,ksize=[1,row,col,1],strides=[1,row,col,1],padding='SAME')

def model_graph(pssm, encoding, domain, keep_prob, y_):
	with tf.name_scope('pretrained_cnn'):
	    with tf.device('/gpu:0'):
	        with tf.name_scope('convnet_for_pssm_feature'):
	            #Reshape the input into 4D Tensor
	            with tf.name_scope('pssm_feature_reshape'):
	                x_pssm=tf.reshape(pssm,[-1,MAX_LENGTH,TYPE_OF_AA,1])

	            #ADD FIRST CONVELUTIONAL LAYER#
	            #add the variables
	            with tf.name_scope('pssm_conv_layer_1'):
	                w_conv1_pssm=weight_variable([40,4,1,32])
	                b_conv1_pssm=bias_variable([32])
	            #add the first conv layer and pooling layer
	                h_conv1_pssm=tf.nn.relu(conv2d(x_pssm,w_conv1_pssm)+b_conv1_pssm)
	                with tf.name_scope('pssm_pool_layer_1'):
	                    h_pool1_pssm=max_pool2d(h_conv1_pssm,5,2)
	                h_pool1_pssm = tflearn.batch_normalization(h_pool1_pssm)

	            #ADD THE SECOND CONVERLUTIONAL LAYER#
	            with tf.name_scope('pssm_conv_layer_2'):
	                w_conv2_pssm=weight_variable([30,4,32,64])
	                b_conv2_pssm=bias_variable([64])
	                h_conv2_pssm=tf.nn.relu(conv2d(h_pool1_pssm,w_conv2_pssm)+b_conv2_pssm)
	                h_conv2_pssm = tflearn.batch_normalization(h_conv2_pssm)

	            #ADD THE THIRD CONVER LAYER AND THE SECOND POOLING LAYER
	            with tf.name_scope('pssm_conv_layer_3'):
	                w_conv3_pssm=weight_variable([30,4,64,128])
	                b_conv3_pssm=bias_variable([128])
	                h_conv3_pssm=tf.nn.relu(conv2d(h_conv2_pssm,w_conv3_pssm)+b_conv3_pssm)
	                with tf.name_scope('pssm_pool_layer_2'):
	                    h_pool2_pssm=max_pool2d(h_conv3_pssm,5,2)
	                h_pool2_pssm = tflearn.batch_normalization(h_pool2_pssm)

	            #ADD THE FORTH CONVER LAYER
	            with tf.name_scope('pssm_conv_layer_4'):
	                w_conv4_pssm=weight_variable([20,3,128,256])
	                b_conv4_pssm=bias_variable([256])
	                h_conv4_pssm=tf.nn.relu(conv2d(h_pool2_pssm,w_conv4_pssm)+b_conv4_pssm)
	                h_conv4_pssm = tflearn.batch_normalization(h_conv4_pssm)

	            #ADD THE FIFTH CONVER LAYER AND THIRD POOLING LAYER
	            with tf.name_scope('pssm_conv_layer_5'):
	                w_conv5_pssm=weight_variable([20,3,256,256])
	                b_conv5_pssm=bias_variable([256])
	                h_conv5_pssm=tf.nn.relu(conv2d(h_conv4_pssm,w_conv5_pssm)+b_conv5_pssm)
	                with tf.name_scope('pssm_pool_layer_3'):
	                    h_pool3_pssm=max_pool2d(h_conv5_pssm,4,1)
	                h_pool3_pssm = tflearn.batch_normalization(h_pool3_pssm)
	            
	            #ADD THE SIXTH CONVER LAYER AND FORTH POOLING LAYER
	            with tf.name_scope('pssm_conv_layer_6'):
	                w_conv6_pssm=weight_variable([20,3,256,256])
	                b_conv6_pssm=bias_variable([256])
	                h_conv6_pssm=tf.nn.relu(conv2d(h_pool3_pssm,w_conv6_pssm)+b_conv6_pssm)
	                with tf.name_scope('pssm_pool_layer_4'):
	                    h_pool4_pssm=max_pool2d(h_conv6_pssm,2,1)
	                h_pool4_pssm = tflearn.batch_normalization(h_pool4_pssm)

	    with tf.device('/gpu:0'):
	        with tf.name_scope('convnet_for_encoding_feature'):
	            #Reshape the input into 4D Tensor
	            with tf.name_scope('encoding_feature_reshape'):
	                x_encoding=tf.reshape(encoding,[-1,1,MAX_LENGTH,1])

	            #ADD FIRST CONVELUTIONAL LAYER#
	            with tf.name_scope('encoding_conv_layer_1'):
	                #add the variables
	                w_conv1_encoding=weight_variable([1,40,1,32])
	                b_conv1_encoding=bias_variable([32])
	                #add the first conv layer and pooling layer
	                h_conv1_encoding=tf.nn.relu(conv2d(x_encoding,w_conv1_encoding)+b_conv1_encoding)
	                with tf.name_scope('encoding_pool_layer_1'):
	                    h_pool1_encoding=max_pool2d(h_conv1_encoding,1,5)
	                h_pool1_encoding = tflearn.batch_normalization(h_pool1_encoding)
	                
	            #ADD THE SECOND CONVERLUTIONAL LAYER#
	            with tf.name_scope('encoding_conv_layer_2'):
	                w_conv2_encoding=weight_variable([1,30,32,64])
	                b_conv2_encoding=bias_variable([64])
	                h_conv2_encoding=tf.nn.relu(conv2d(h_pool1_encoding,w_conv2_encoding)+b_conv2_encoding)
	                h_conv2_encoding = tflearn.batch_normalization(h_conv2_encoding)

	            #ADD THE THIRD CONVER LAYER AND THE SECOND POOLING LAYER
	            with tf.name_scope('encoding_conv_layer_3'):
	                w_conv3_encoding=weight_variable([1,30,64,128])
	                b_conv3_encoding=bias_variable([128])
	                h_conv3_encoding=tf.nn.relu(conv2d(h_conv2_encoding,w_conv3_encoding)+b_conv3_encoding)
	                with tf.name_scope('encoding_pool_layer_2'):
	                    h_pool2_encoding=max_pool2d(h_conv3_encoding,1,5)
	                h_pool2_encoding = tflearn.batch_normalization(h_pool2_encoding)
	                
	            #ADD THE FORTH CONVER LAYER
	            with tf.name_scope('encoding_conv_layer_4'):
	                w_conv4_encoding=weight_variable([1,20,128,256])
	                b_conv4_encoding=bias_variable([256])
	                h_conv4_encoding=tf.nn.relu(conv2d(h_pool2_encoding,w_conv4_encoding)+b_conv4_encoding)
	                h_conv4_encoding = tflearn.batch_normalization(h_conv4_encoding)

	            #ADD THE FIFTH CONVER LAYER AND THIRD POOLING LAYER
	            with tf.name_scope('encoding_conv_layer_5'):
	                w_conv5_encoding=weight_variable([1,20,256,256])
	                b_conv5_encoding=bias_variable([256])
	                h_conv5_encoding=tf.nn.relu(conv2d(h_conv4_encoding,w_conv5_encoding)+b_conv5_encoding)
	                with tf.name_scope('encoding_pool_layer_3'):
	                    h_pool3_encoding=max_pool2d(h_conv5_encoding,1,4)
	                h_pool3_encoding = tflearn.batch_normalization(h_pool3_encoding)
	                
	            #ADD THE SIXTH CONVER LAYER AND FORTH POOLING LAYER
	            with tf.name_scope('encoding_conv_layer_6'):
	                w_conv6_encoding=weight_variable([1,20,256,256])
	                b_conv6_encoding=bias_variable([256])
	                h_conv6_encoding=tf.nn.relu(conv2d(h_pool3_encoding,w_conv6_encoding)+b_conv6_encoding)
	                with tf.name_scope('encoding_pool_layer_4'):
	                    h_pool4_encoding=max_pool2d(h_conv6_encoding,1,2)
	                h_pool4_encoding = tflearn.batch_normalization(h_pool4_encoding)

	#consturct dimensionality redution for functional domain encoding
	with tf.name_scope('fine_tune_layers'):
	    with tf.name_scope('functional_domain_layers'):
	        with tf.name_scope('functional_domain_fc_1'):
	            w_dr1_domain=weight_variable([DOMAIN,4096])
	            b_dr1_domain=bias_variable([4096])
	            h_dr1_domain=tflearn.prelu(tf.matmul(domain,w_dr1_domain)+b_dr1_domain)
	            h_dr1_domain=tflearn.batch_normalization(h_dr1_domain)

	        with tf.name_scope('functional_domain_fc_2'):
	            w_dr2_domain=weight_variable([4096,1024])
	            b_dr2_domain=bias_variable([1024])
	            h_dr2_domain=tflearn.prelu(tf.matmul(h_dr1_domain,w_dr2_domain)+b_dr2_domain)
	            h_dr2_domain=tflearn.batch_normalization(h_dr2_domain)

	    #ADD THE DENSELY CONNECTED LAYER#
	    with tf.name_scope('densely_connected_layers'):
	        with tf.name_scope('fc_1'):
	            b_fc1=bias_variable([1024])
	            w_fc1_pssm=weight_variable([25*5*256,1024])
	            h_pool4_pssm_flat=tf.reshape(h_pool4_pssm,[-1,25*5*256])
	            w_fc1_encoding=weight_variable([1*25*256,1024])
	            h_pool4_encoding_flat=tf.reshape(h_pool4_encoding,[-1,1*25*256])
	            #incoporate functional domain encoding information
	            w_fc1_domain=weight_variable([1024,1024])
	            h_fc1=tf.nn.relu(
	            	tf.matmul(h_pool4_pssm_flat,w_fc1_pssm)+
	            	tf.matmul(h_pool4_encoding_flat,w_fc1_encoding)+
	            	tf.matmul(h_dr2_domain,w_fc1_domain)+b_fc1)
	            h_fc1=tflearn.batch_normalization(h_fc1)
	            if DROPOUT==True:
	                h_fc1=tf.nn.dropout(h_fc1,keep_prob)

	    with tf.name_scope('fc_3'):
	        #Add the third densely connected layer
	        w_fc3=weight_variable([1024,1024])
	        b_fc3=bias_variable([1024])
	        h_fc3=tflearn.prelu(tf.matmul(h_fc1,w_fc3)+b_fc3)
	        h_fc3=tflearn.batch_normalization(h_fc3)
	        if DROPOUT==True:
	            h_fc3=tf.nn.dropout(h_fc3,keep_prob)

	    #ADD SOFTMAX LAYER
	    with tf.name_scope('softmax_layer'):
	        w_fc4=weight_variable([1024,y_.get_shape().as_list()[1]])
	        b_fc4=bias_variable([y_.get_shape().as_list()[1]])
	        y_conv_logit=tf.matmul(h_fc3,w_fc4)+b_fc4
	        y_conv=tf.nn.softmax(y_conv_logit)
	return y_conv_logit, y_conv, h_fc3

