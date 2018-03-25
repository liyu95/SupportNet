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
from level_2_preprocess import exclude_data
from evaluate_model import *
import pdb
from level_1_model_graph import model_graph
from utils import *
import argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

start=time.time()

#Global variable, MAX_LENGTH is the maximum length of all sequence.
DROPOUT=True
MAX_LENGTH=5000
TYPE_OF_AA=20
DOMAIN=16306
LOAD=False
train_ratio=0.9
level=1
output_step=100
batch_size = 20
train_steps = 500

#load all data
data_all = load_level_1_data(level)
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

y_conv_logit, y_conv, final_feature = model_graph(pssm, encoding, domain, keep_prob, y_)

#DEFINE LOSS FUNCTION
with tf.name_scope('cross_entropy'):
	cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		labels=y_, logits=y_conv_logit))
	tf.summary.scalar('cross_entropy',cross_entropy)

pretrain_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'pretrained_cnn')
fine_tune_collection=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'fine_tune_layers')
softmax_collection = filter(lambda x: 'softmax_layer' in x.name, fine_tune_collection)
var_list = pretrain_collection + fine_tune_collection

with tf.name_scope('train'):
    global_step=tf.Variable(0,trainable=False)
    starter_learning_rate=0.01
    learning_rate=tf.train.exponential_decay(starter_learning_rate,global_step,200,0.96,staircase=True)
    weight_collection=[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weight_collection])
    theta_1 = 0.0001
    theta_2 = 0.001
    cross_entropy_with_weight_decay=tf.add(cross_entropy,theta_1*l2_loss)
    train_op=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_with_weight_decay, var_list=softmax_collection)


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
saver_load = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))


def whole_set_check_simple(data):
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
    # number_of_full_batch=int(math.floor(len(train_label)/batch_size))
    # for i in range(number_of_full_batch):
    #     predicted_label_out,fea_out = sess.run([predicted_label,final_feature],
    #         feed_dict={pssm: train_pssm[i*batch_size:(i+1)*batch_size], 
    #         encoding: train_encoding[i*batch_size:(i+1)*batch_size], 
    #         domain: train_funcd[i*batch_size:(i+1)*batch_size], 
    #         y_: train_label_c[i*batch_size:(i+1)*batch_size], keep_prob: 1.0})
    #     final_train_fea+=list(fea_out)
    #     predict_train_label+=list(predicted_label_out)

    # predicted_label_out,fea_out = sess.run([predicted_label,final_feature],
    #     feed_dict={pssm: train_pssm[number_of_full_batch*batch_size:], 
    #     encoding: train_encoding[number_of_full_batch*batch_size:], 
    #     domain: train_funcd[number_of_full_batch*batch_size:], 
    #     y_: train_label_c[number_of_full_batch*batch_size:], keep_prob: 1.0})
    # final_train_fea+=list(fea_out)
    # predict_train_label+=list(predicted_label_out)

    final_test_fea=[]
    predict_test_label=[]
    number_of_full_batch=int(math.floor(len(test_label)/batch_size))
    for i in range(number_of_full_batch):
        predicted_label_out,fea_out = sess.run([predicted_label,final_feature],
            feed_dict={pssm: test_pssm[i*batch_size:(i+1)*batch_size], 
            encoding: test_encoding[i*batch_size:(i+1)*batch_size], 
            domain: test_funcd[i*batch_size:(i+1)*batch_size], 
            y_: test_label_c[i*batch_size:(i+1)*batch_size], keep_prob: 1.0})
        final_test_fea+=list(fea_out)
        predict_test_label+=list(predicted_label_out)
    
    predicted_label_out,fea_out = sess.run([predicted_label,final_feature],
        feed_dict={pssm: test_pssm[number_of_full_batch*batch_size:], 
        encoding: test_encoding[number_of_full_batch*batch_size:], 
        domain: test_funcd[number_of_full_batch*batch_size:], 
        y_: test_label_c[number_of_full_batch*batch_size:], keep_prob: 1.0})
    
    final_test_fea+=list(fea_out)
    predict_test_label+=list(predicted_label_out)

    return (predict_test_label, final_test_fea)
    # return (predict_train_label,predict_test_label, final_train_fea, final_test_fea)

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
    number_of_full_batch=int(math.floor(len(train_label)/batch_size))
    for i in range(number_of_full_batch):
        predicted_label_out,fea_out = sess.run([predicted_label,final_feature
            # tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv_logit)
            ],
            feed_dict={pssm: train_pssm[i*batch_size:(i+1)*batch_size], 
            encoding: train_encoding[i*batch_size:(i+1)*batch_size], 
            domain: train_funcd[i*batch_size:(i+1)*batch_size], 
            y_: train_label_c[i*batch_size:(i+1)*batch_size], keep_prob: 1.0})
        final_train_fea+=list(fea_out)
        predict_train_label+=list(predicted_label_out)

    predicted_label_out,fea_out = sess.run([predicted_label,final_feature
        # tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv_logit)
        ],
        feed_dict={pssm: train_pssm[number_of_full_batch*batch_size:], 
        encoding: train_encoding[number_of_full_batch*batch_size:], 
        domain: train_funcd[number_of_full_batch*batch_size:], 
        y_: train_label_c[number_of_full_batch*batch_size:], keep_prob: 1.0})
    final_train_fea+=list(fea_out)
    predict_train_label+=list(predicted_label_out)
    # print("Whole set train accuracy %g"%(sum(final_train_acc)/float(len(final_train_acc))))

    final_test_fea=[]
    predict_test_label=[]
    number_of_full_batch=int(math.floor(len(test_label)/batch_size))
    for i in range(number_of_full_batch):
        predicted_label_out,fea_out = sess.run([predicted_label,final_feature],
            feed_dict={pssm: test_pssm[i*batch_size:(i+1)*batch_size], 
            encoding: test_encoding[i*batch_size:(i+1)*batch_size], 
            domain: test_funcd[i*batch_size:(i+1)*batch_size], 
            y_: test_label_c[i*batch_size:(i+1)*batch_size], keep_prob: 1.0})
        final_test_fea+=list(fea_out)
        predict_test_label+=list(predicted_label_out)
    
    predicted_label_out,fea_out = sess.run([predicted_label,final_feature],
        feed_dict={pssm: test_pssm[number_of_full_batch*batch_size:], 
        encoding: test_encoding[number_of_full_batch*batch_size:], 
        domain: test_funcd[number_of_full_batch*batch_size:], 
        y_: test_label_c[number_of_full_batch*batch_size:], keep_prob: 1.0})
    
    final_test_fea+=list(fea_out)
    predict_test_label+=list(predicted_label_out)
    # print("Whole set test accuracy %g"%(sum(final_test_acc)/float(len(final_test_acc))))
    return (predict_train_label,predict_test_label, final_train_fea, final_test_fea)

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
            predict_label,_=whole_set_check_simple(data)
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

def check_fix_rep_batch_performance(train_op,exclude_list, data_merged):
    used_data = exclude_data(data_all, exclude_list)
    data_merged = merge_data(used_data, data_merged)

    train_model(used_data, train_op, train_steps)
    predict_label_train,predict_label,final_train_fea,final_test_fea=whole_set_check(data_merged)
    evaluate_model(data_merged[5],np.array(predict_label))
    evaluate_model(data_merged[0],np.array(predict_label_train))


def restart_from_ckpt(s_class):
    if s_class==1:
        data_1 = exclude_data(data_all, range(2,6))
        saver_load.restore(sess, '../model/fsize_100_flam_1000_ssize_2000_class_2.ckpt')
        # check performance
        predict_label_train,predict_label,final_train_fea,final_test_fea=whole_set_check(data_1)
        evaluate_model(data_1[5],np.array(predict_label))
        evaluate_model(data_1[0],np.array(predict_label_train))

    if s_class>=2:
        print('load model: fix_rep_class_{}.ckpt'.format(s_class))
        saver_load.restore(sess,'../model/fix_rep_class_{}.ckpt'.format(s_class))

        data_1 = exclude_data(data_all, range(s_class,6))

        train_loss, data_merged = check_fix_rep_batch_performance(train_op,
            range(s_class)+range(s_class+1,6), data_1)

    saver_load.save(sess,'../model/fix_rep_class_{}.ckpt'.format(s_class+1))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='input argument')
    # parser.add_argument('-s', action='store', dest='start', type=int,
    #     help='the start class')
    # parser.add_argument('-l', action='store', dest='lam', type=float,
    #     help='the coefficient for the regularizer')
    # args = parser.parse_args()
    for i in range(1,6):
        restart_from_ckpt(i)

