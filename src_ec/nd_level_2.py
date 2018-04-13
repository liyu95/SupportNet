import time
import numpy as np
import tflearn
from sklearn.cross_validation import train_test_split
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
from level_2_preprocess import load_level_2_data, exclude_data
from evaluate_model import *
import pdb
from level_2_model_graph import model_graph
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

start=time.time()

#Global variable, MAX_LENGTH is the maximum length of all sequence.
DROPOUT=True
MAX_LENGTH=5000
TYPE_OF_AA=20
DOMAIN=16306
LOAD=True
train_ratio=0.9
level=2
n_class=1
output_step=100
batch_size = 20
train_steps = 1500

#load all data
data_all = load_level_2_data(level, n_class)
all_train_label = data_all[0]
num_total_class = len(set(data_all[0]))

config = tf.ConfigProto()
config.log_device_placement=False
config.allow_soft_placement=True
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

with tf.name_scope('placeholder'):
    pssm=tf.placeholder(tf.float32,shape=[None,MAX_LENGTH,TYPE_OF_AA])
    encoding=tf.placeholder(tf.float32,shape=[None,MAX_LENGTH])
    y_=tf.placeholder(tf.float32,shape=[None, num_total_class])
    domain=tf.placeholder(tf.float32,shape=[None,DOMAIN])
    keep_prob=tf.placeholder(tf.float32)

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
            w_dr1_domain=weight_variable([DOMAIN,1024])
            b_dr1_domain=bias_variable([1024])
            h_dr1_domain=tf.nn.relu(tf.matmul(domain,w_dr1_domain)+b_dr1_domain)
            h_dr1_domain=tflearn.batch_normalization(h_dr1_domain)

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
            h_fc1=tf.nn.relu(tf.matmul(h_pool4_pssm_flat,w_fc1_pssm)+
            	tf.matmul(h_pool4_encoding_flat, w_fc1_encoding)+
            	tf.matmul(h_dr1_domain,w_fc1_domain)+
            	b_fc1)
            h_fc1=tflearn.batch_normalization(h_fc1)
            if DROPOUT==True:
                h_fc1=tf.nn.dropout(h_fc1,keep_prob)

    #ADD SOFTMAX LAYER
    with tf.name_scope('softmax_layer'):
        w_fc4=weight_variable([1024,y_.get_shape().as_list()[1]])
        b_fc4=bias_variable([y_.get_shape().as_list()[1]])
        y_conv_logit=tf.matmul(h_fc1,w_fc4)+b_fc4
        y_conv=tf.nn.softmax(y_conv_logit)
        ##normal softmax end

#DEFINE LOSS FUNCTION
with tf.name_scope('cross_entropy'):
	cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		labels=y_, logits=y_conv_logit))
	tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope('train'):
    global_step=tf.Variable(0,trainable=False)
    starter_learning_rate=0.01
    learning_rate=tf.train.exponential_decay(starter_learning_rate,global_step,200,0.96,staircase=True)
    weight_collection=[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weight_collection])
    theta_1 = 0.0001
    theta_2 = 0.001
    # fine_tune_collection=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'fine_tune_layers')
    # pretrain_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'pretrained_cnn')
    # pretrain_coll_values = [v.eval() for v in pretrain_collection]
    # incre_loss = tf.add_n([tf.nn.l2_loss(pretrain_collection[i]-pretrain_coll_values[i]) 
    #     for i in range(len(pretrain_collection))])
    cross_entropy_with_weight_decay=tf.add(cross_entropy,theta_1*l2_loss)
    #train_step=tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(
    	# cross_entropy_with_weight_decay,global_step=global_step)
    train_op=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_with_weight_decay)


pretrain_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'pretrained_cnn')
fine_tune_collection=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'fine_tune_layers')
var_list = pretrain_collection + fine_tune_collection
#DEFINE EVALUATION
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        actual_label=tf.argmax(y_,1)
        predicted_label=tf.argmax(y_conv,1)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy',accuracy)

merged=tf.summary.merge_all()
if not os.path.isdir('./log_dir_level_'+str(level)):
    os.system('mkdir log_dir_level_'+str(level))
    os.system('mkdir log_dir_level_'+str(level)+'/train')
    os.system('mkdir log_dir_level_'+str(level)+'/test')

train_writer=tf.summary.FileWriter('./log_dir_level_'+str(level)+'/train/',sess.graph)
test_writer=tf.summary.FileWriter('./log_dir_level_'+str(level)+'/test/')

sess.run(tf.global_variables_initializer())
saver_load = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'pretrained_cnn'))
saver = tf.train.Saver()

def adam_variables_initializer(adam_opt, var_list):
    adam_vars = [adam_opt.get_slot(var, name)
                 for name in adam_opt.get_slot_names()
                 for var in var_list if var is not None]
    adam_vars.extend(list(adam_opt._get_beta_accumulators()))
    return tf.variables_initializer(adam_vars)

# the initalization for the feature layers
if LOAD==True:
    saver_load.restore(sess, '../model/model_level_'+str(level)+'.ckpt')
    print('Model load successful!')

#function to generate feed batch
def generate_feeding_batch(pssm,encoding,domain,train_label,batch_size):
    from numpy.random import randint
    batch_index=randint(0,len(pssm),batch_size)
    domain_batch=[]
    pssm_batch=[]
    encoding_batch=[]
    label_batch=[]
    for index in batch_index:
        domain_batch.append(domain[index])
        pssm_batch.append(pssm[index])
        encoding_batch.append(encoding[index])
        label_batch.append(train_label[index])
    pssm_batch=np.array(pssm_batch)
    encoding_batch=np.array(encoding_batch)
    label_batch=np.array(label_batch)
    domain_batch=np.array(domain_batch)
    pssm_batch=pssm_batch.astype('float')
    encoding_batch=encoding_batch.astype('float')
    label_batch=label_batch.astype('float')
    domain_batch=domain_batch.astype('float')
    return (pssm_batch,encoding_batch,domain_batch,label_batch)

def whole_set_check(data):
    train_label = data[0]
    train_label_c = data[1]
    train_pssm = data[2]
    train_encoding = data[3]
    train_funcd =data[4]
    test_label = data[5]
    test_label_c = data[6]
    test_pssm = data[7]
    test_encoding = data[8]
    test_funcd = data[9]
    final_train_fea=[]
    predict_train_label=[]
    train_loss = []
    number_of_full_batch=int(math.floor(len(train_label)/batch_size))
    for i in range(number_of_full_batch):
        predicted_label_out,fea_out, loss_out = sess.run([predicted_label,h_fc1, 
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv_logit)],
            feed_dict={pssm: train_pssm[i*batch_size:(i+1)*batch_size], 
            encoding: train_encoding[i*batch_size:(i+1)*batch_size], 
            domain: train_funcd[i*batch_size:(i+1)*batch_size], 
            y_: train_label_c[i*batch_size:(i+1)*batch_size], keep_prob: 1.0})
        final_train_fea+=list(fea_out)
        predict_train_label+=list(predicted_label_out)
        train_loss+=list(loss_out)

    predicted_label_out,fea_out, loss_out = sess.run([predicted_label,h_fc1,
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv_logit)],
        feed_dict={pssm: train_pssm[number_of_full_batch*batch_size:], 
        encoding: train_encoding[number_of_full_batch*batch_size:], 
        domain: train_funcd[number_of_full_batch*batch_size:], 
        y_: train_label_c[number_of_full_batch*batch_size:], keep_prob: 1.0})
    final_train_fea+=list(fea_out)
    predict_train_label+=list(predicted_label_out)
    train_loss+=list(loss_out)
    # print("Whole set train accuracy %g"%(sum(final_train_acc)/float(len(final_train_acc))))

    final_test_fea=[]
    predict_test_label=[]
    number_of_full_batch=int(math.floor(len(test_label)/batch_size))
    for i in range(number_of_full_batch):
        predicted_label_out,fea_out = sess.run([predicted_label,h_fc1],
            feed_dict={pssm: test_pssm[i*batch_size:(i+1)*batch_size], 
            encoding: test_encoding[i*batch_size:(i+1)*batch_size], 
            domain: test_funcd[i*batch_size:(i+1)*batch_size], 
            y_: test_label_c[i*batch_size:(i+1)*batch_size], keep_prob: 1.0})
        final_test_fea+=list(fea_out)
        predict_test_label+=list(predicted_label_out)
    
    predicted_label_out,fea_out = sess.run([predicted_label,h_fc1],
        feed_dict={pssm: test_pssm[number_of_full_batch*batch_size:], 
        encoding: test_encoding[number_of_full_batch*batch_size:], 
        domain: test_funcd[number_of_full_batch*batch_size:], 
        y_: test_label_c[number_of_full_batch*batch_size:], keep_prob: 1.0})
    
    final_test_fea+=list(fea_out)
    predict_test_label+=list(predicted_label_out)
    # print("Whole set test accuracy %g"%(sum(final_test_acc)/float(len(final_test_acc))))

    return (train_loss, predict_train_label,predict_test_label, final_train_fea, final_test_fea)
# Training function can accept different training op and train steps
def train_model(data, train_op, train_steps):
    train_label = data[0]
    train_label_categorical = data[1]
    train_pssm_full_length = data[2]
    train_encoding_full_length = data[3]
    train_functional_domain_encoding =data[4]
    test_label = data[5]
    test_label_categorical = data[6]
    test_pssm_full_length = data[7]
    test_encoding_full_length = data[8]
    test_functional_domain_encoding = data[9]
    for i in range(train_steps):
        batch=generate_feeding_batch(train_pssm_full_length,train_encoding_full_length,
        	train_functional_domain_encoding,train_label_categorical,batch_size)
        if i%output_step == 0:
            summary,predicted_label_output,cross_entropy_output,y_conv_output,acc = sess.run([
            	merged, predicted_label,cross_entropy,y_conv,accuracy],feed_dict={pssm: batch[0], 
                encoding: batch[1], domain: batch[2], y_: batch[3], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, acc))
            batch_test=generate_feeding_batch(test_pssm_full_length,test_encoding_full_length,
            	test_functional_domain_encoding,test_label_categorical,batch_size)
            summary,acc= sess.run([merged,accuracy],feed_dict={
            	pssm: batch_test[0], 
            	encoding: batch_test[1], 
            	domain: batch_test[2], 
            	y_: batch_test[3], 
            	keep_prob: 1.0})
            test_writer.add_summary(summary,i)
            print("step %d, test accuracy %g"%(i, acc))
            print('cross_entropy: %g'%cross_entropy_output)
            print(predicted_label_output)
        if i%1000==0 and i!=0:
            print('Step %d whole set check'%i)
            _, _,predict_label,_,_=whole_set_check(data)
            evaluate_model(test_label[:len(predict_label)],np.array(predict_label))
        # if i%2000==0:
        #     save_path=saver.save(sess,'./model_level_'+str(level)+'.ckpt')
        summary,cross_entropy_output,_ = sess.run([merged,cross_entropy,train_op],feed_dict={
        	pssm: batch[0], 
            encoding: batch[1], 
            domain: batch[2], 
            y_: batch[3], 
            keep_prob: 0.5})
        train_writer.add_summary(summary,i)

# incremental learning, every time use all the data
def incremental_all_data():
    # get partial data
    data_1 = exclude_data(data_all, range(5,21))

    train_model(data_1, train_op, train_steps)

    # check performance
    train_loss, predict_label_train,predict_label,final_train_fea,final_test_fea=whole_set_check(data_1)
    evaluate_model(data_1[5],np.array(predict_label))
    evaluate_model(data_1[0],np.array(predict_label_train))

    # get further data
    data_2 = exclude_data(data_all, range(5)+range(10,21))
    data_merged = merge_data(data_1, data_2)
    train_model(data_merged, train_op, train_steps)

    train_loss, predict_label_train,predict_label,final_train_fea,final_test_fea=whole_set_check(data_merged)
    evaluate_model(data_merged[5],np.array(predict_label))
    evaluate_model(data_merged[0],np.array(predict_label_train))

    # get further data
    data_2 = exclude_data(data_all, range(10)+range(15,21))
    data_merged = merge_data(data_merged, data_2)
    train_model(data_merged, train_op, train_steps)

    train_loss, predict_label_train,predict_label,final_train_fea,final_test_fea=whole_set_check(data_merged)
    evaluate_model(data_merged[5],np.array(predict_label))
    evaluate_model(data_merged[0],np.array(predict_label_train))

    # get further data
    data_2 = exclude_data(data_all, range(15))
    data_merged = merge_data(data_merged, data_2)
    train_model(data_merged, train_op, train_steps)

    train_loss, predict_label_train,predict_label,final_train_fea,final_test_fea=whole_set_check(data_merged)
    evaluate_model(data_merged[5],np.array(predict_label))
    evaluate_model(data_merged[0],np.array(predict_label_train))

def check_support_data_batch_performance(train_op,exclude_list, train_loss, 
    data_merged, support_size, train_steps, aug_rate):
    # construct the support data for the first and second batch
    support_data_index = select_support_data(train_loss, data_merged[0], support_size, all_train_label)
    support_data = get_support_data(data_merged, support_data_index)

    # load the second batch data
    data_2 = exclude_data(data_all, exclude_list)
    data_merged = merge_data(support_data, data_2)
    used_data = merge_data(augmentation(support_data, aug_rate), data_2)
    train_model(used_data, train_op, train_steps)

    train_loss, predict_label_train,predict_label,final_train_fea,final_test_fea=whole_set_check(data_merged)
    evaluate_model(data_merged[5],np.array(predict_label))
    evaluate_model(data_merged[0],np.array(predict_label_train))

    return train_loss, data_merged

# during training, only use the support data and the current batch data
def support_data_without_ewc(train_op):
    # get partial data
    data_1 = exclude_data(data_all, range(5,21))

    train_model(data_1, train_op, train_steps)

    # check performance
    train_loss, predict_label_train,predict_label,final_train_fea,final_test_fea=whole_set_check(data_1)
    evaluate_model(data_1[5],np.array(predict_label))
    evaluate_model(data_1[0],np.array(predict_label_train))

    train_loss, data_merged = check_support_data_batch_performance(train_op,range(5)+range(10,21), 
        train_loss, data_1, 400, 1000, 1)

    train_loss, data_merged = check_support_data_batch_performance(train_op,range(10)+range(15,21), 
        train_loss, data_merged, 400, 1000, 2)

    train_loss, data_merged = check_support_data_batch_performance(train_op,range(15), 
        train_loss, data_merged, 400, 1000, 3)

def calculate_fish(data, var_list, num_samples):
    # calcuate fisher information
    train_label = data[0]
    train_label_c = data[1]
    train_pssm = data[2]
    train_encoding = data[3]
    train_funcd =data[4]
    test_label = data[5]
    test_label_c = data[6]
    test_pssm = data[7]
    test_encoding = data[8]
    test_funcd = data[9]
    F_accum = []
    for v in range(len(var_list)):
        F_accum.append(np.zeros(var_list[v].get_shape().as_list()))
    class_ind = tf.to_int32(tf.multinomial(tf.log(y_conv), 1)[0][0])    
    for i in range(num_samples):
        if i%10==0:
            print(i)
        # select random input
        ind = np.random.randint(len(test_label))
        # compute first-order derivatives
        ders = sess.run(tf.gradients(tf.log(y_conv[0,class_ind]), var_list), 
            feed_dict={pssm: test_pssm[ind:ind+1], 
            encoding: test_encoding[ind:ind+1], 
            domain: test_funcd[ind:ind+1], 
            y_: test_label_c[ind:ind+1], keep_prob: 1.0})
        # square the derivatives and add to total
        for v in range(len(F_accum)):
            F_accum[v] += np.square(ders[v])

    # divide totals by number of samples
    for v in range(len(F_accum)):
        F_accum[v] /= num_samples
    return F_accum

def set_fisher_regularizer(lam, data, iteration):
    var_trained_list = [v.eval() for v in var_list]

    F = calculate_fish(data, var_list,iteration)
    ewc_loss  = 0
    for v in range(len(var_list)):
        ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(F[v].astype(np.float32),
            tf.square(var_list[v] - var_trained_list[v])))
    incre_loss = cross_entropy + ewc_loss
    opt = tf.train.AdamOptimizer(learning_rate=1e-4)

    # only reset the variable related to the optimizer
    new_step=opt.minimize(incre_loss, var_list = var_list)
    reset_opt_vars = adam_variables_initializer(opt, var_list)
    sess.run(reset_opt_vars)

    return new_step


# the final version useing both the regularizer and the support data
def support_with_regularizer():
    pass

if __name__ == '__main__':
    # get partial data
    data_1 = exclude_data(data_all, range(5,21))

    train_model(data_1, train_op, train_steps)

    # check performance
    train_loss, predict_label_train,predict_label,final_train_fea,final_test_fea=whole_set_check(data_1)
    evaluate_model(data_1[5],np.array(predict_label))
    evaluate_model(data_1[0],np.array(predict_label_train))

    # set the regularizer
    new_step = set_fisher_regularizer(2, data_1, 50)
    train_loss, data_merged = check_support_data_batch_performance(new_step,range(5)+range(10,21), 
        train_loss, data_1, 400, 1000, 1)

    new_step = set_fisher_regularizer(2, data_merged, 50)
    train_loss, data_merged = check_support_data_batch_performance(train_op,range(10)+range(15,21), 
        train_loss, data_merged, 400, 800, 1)

    new_step = set_fisher_regularizer(2, data_merged, 50)
    train_loss, data_merged = check_support_data_batch_performance(train_op,range(15), 
        train_loss, data_merged, 400, 700, 1)


    # have a test to see if the new op can run or not


    # test_pred_knn = nme_pred(final_train_fea, final_test_fea, train_label)
    # evaluate_model(test_label, test_pred_knn)

    #save model
    # save_path=saver.save(sess,'./model_level_'+str(level)+'.ckpt')

    end=time.time()
    print("Running time %d min"%((end-start)/60))
