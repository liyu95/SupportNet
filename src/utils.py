import tensorflow as tf
import numpy as np
from collections import Counter
import tflearn
from sklearn.cross_validation import train_test_split
import random
import cPickle
import copy
import math
import sys
sys.path.append('/home/liy0f/ec_project/data_and_feature')
import protein_sequence_process_functions as p_func
import Pfam_pickle_file_to_array_encoding as Pfam
import os
from level_2_preprocess import label_one_hot, construct_feature_dictionary
from level_2_preprocess import construct_label_dictionary, train_test_data_generation
from level_2_preprocess import encoding_to_1d
from sklearn import svm


#Global variable, MAX_LENGTH is the maximum length of all sequence.
MAX_LENGTH=5000
TYPE_OF_AA=20
DOMAIN=16306
train_ratio=0.9
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

def load_level_1_data(level=1):
    f=open('/home/liy0f/ec_project/data_and_feature/new_data_label_sequence.txt','r')
    text=f.read()
    f.close()
    #using "\n" to convert the data into list
    sequence_list=text.split('\n')
    #The last element of sequence_list is '', remove it
    sequence_list.pop(-1)
    #load PSSM dataset
    f=open('/home/liy0f/ec_project/data_and_feature/PSSM_new_data_first_matrix_list.pickle','r')
    #The length of each pssm profile is the same as the length of the sequence, which could not be a formal input of
    #a neural network
    pssm_list_original_length=cPickle.load(f)
    f.close()

    #load sequence encoding dataset
    f=open('/home/liy0f/ec_project/data_and_feature/seqence_encoded_array_original_length_new_data.pickle','r')
    sequence_encoding_list_original_length=cPickle.load(f)
    f.close()

    #load Pfam dataset and encoding them into array
    functional_domain_encoding_list=Pfam.Pfam_from_pickle_file_encoding(
        '/home/liy0f/ec_project/data_and_feature/Pfam_name_list_new_data.pickle',
        '/home/liy0f/ec_project/data_and_feature/Pfam_model_names_list.pickle')

    #construct feature dictionary for different features
    pssm_dictionary_original_length=construct_feature_dictionary(sequence_list,pssm_list_original_length)
    encoding_dictionary_original_length=construct_feature_dictionary(sequence_list,sequence_encoding_list_original_length)
    functional_domain_encoding_dictionary=construct_feature_dictionary(sequence_list,functional_domain_encoding_list)

    #construct sequence label dictionary
    label_dictionary=construct_label_dictionary(sequence_list, level)

    label_list=[]
    for i in range(len(sequence_list)):
        label_list.append(label_dictionary[sequence_list[i].split('>')[1]])

    data_size=len(sequence_list)

    #Dropout the sequence that is longer than MAXIMUM LENGTH
    for i in range(len(sequence_list)-1,-1,-1):
        if len(sequence_list[i])>(MAX_LENGTH+1):
            sequence_list.pop(i)

    #Train sequence and test sequence generation
    random_seed=6
    train_sequence,test_sequence=train_test_data_generation(sequence_list,label_list,train_ratio,random_seed)

    #generate label array corresponding to the train and test sequence
    train_label=p_func.label_array_generation(train_sequence,label_dictionary,level)
    test_label=p_func.label_array_generation(test_sequence,label_dictionary,level)

    train_label = train_label-1
    test_label = test_label -1
    unique_label=list(set(label_list))
    unique_label.sort()

    #Convert the label matrix into one-hot from
    train_label_categorical,test_label_categorical=label_one_hot(train_label,test_label,len(unique_label))

    #feature array generation
    train_pssm_full_length=p_func.feature_array_generation(train_sequence,pssm_dictionary_original_length,MAX_LENGTH)
    test_pssm_full_length=p_func.feature_array_generation(test_sequence,pssm_dictionary_original_length,MAX_LENGTH)

    train_encoding_full_length=p_func.feature_array_generation(train_sequence,encoding_dictionary_original_length,MAX_LENGTH)
    test_encoding_full_length=p_func.feature_array_generation(test_sequence,encoding_dictionary_original_length,MAX_LENGTH)

    train_encoding_full_length=encoding_to_1d(train_encoding_full_length)
    test_encoding_full_length=encoding_to_1d(test_encoding_full_length)

    train_functional_domain_encoding=p_func.feature_array_generation(train_sequence,functional_domain_encoding_dictionary)
    test_functional_domain_encoding=p_func.feature_array_generation(test_sequence,functional_domain_encoding_dictionary)
    return (train_label, train_label_categorical, train_pssm_full_length, train_encoding_full_length, 
        train_functional_domain_encoding, test_label, test_label_categorical, test_pssm_full_length,
        test_encoding_full_length, test_functional_domain_encoding)

def adam_variables_initializer(adam_opt, var_list):
    adam_vars = [adam_opt.get_slot(var, name)
                 for name in adam_opt.get_slot_names()
                 for var in var_list if var is not None]
    adam_vars.extend(list(adam_opt._get_beta_accumulators()))
    return tf.variables_initializer(adam_vars)

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

def nme_pred(train_fea, test_fea, train_label):
    from sklearn.neighbors import KNeighborsClassifier
    train_fea = np.array(train_fea)
    test_fea = np.array(test_fea)
    train_label = np.array(train_label)
    unique_label = sorted(set(train_label))
    class_mean_fea = list()
    for label in unique_label:
        class_fea = np.mean(train_fea[train_label==label], 0)
        class_mean_fea.append(class_fea)
    class_mean_fea = np.array(class_mean_fea)
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(class_mean_fea, unique_label)
    test_pred = neigh.predict(test_fea)
    return test_pred


def merge_data(data_1, data_2):
    data_merged = list()
    for i in range(min(len(data_1), len(data_2))):
        temp = np.concatenate((data_1[i], data_2[i]), axis=0)
        data_merged.append(temp)
    return data_merged

# select the data around the median of the loss result
# ratio: the ratio before median
def select_around_median(loss, size, ratio=0.8):
    original_index = np.argsort(loss)
    median_index = int(len(original_index)/2)
    before_median = int(size*ratio)
    after_median = int(size - before_median)
    return original_index[median_index-before_median:median_index+after_median]

def select_support_data(loss, label, total_size, all_label, ratio=0.8):
    label_set = set(label)
    all_related_label = filter(lambda x: x in label_set, all_label)
    sample_ratio = float(total_size)/float(len(label))
    final_index = list()
    loss = np.array(loss)
    for l in label_set:
        # print(l)
        index_specific_label = np.where(label==l)[0]
        loss_specific_label = loss[index_specific_label]
        selected_index = select_around_median(loss_specific_label,
            np.round(len(index_specific_label)*sample_ratio), ratio)
        original_index = index_specific_label[selected_index]
        final_index += list(original_index)
    return final_index


def select_support_data_svm(train_fea, label, total_size,ratio=0.8):
    label_set = set(label)
    train_fea = np.array(train_fea)
    clf = svm.SVC(verbose=1)
    clf.fit(train_fea, label)
    final_index = list(clf.support_)
    sample_ratio = float(total_size-len(final_index))/float(len(label))
    if sample_ratio>0:
        for l in label_set:
            # print(l)
            index_specific_label = np.where(label==l)[0]
            selected_index = np.random.choice(index_specific_label,
                int(np.round(len(index_specific_label)*sample_ratio)),
                replace=False)
            final_index += list(set(list(selected_index)))
    return final_index


def get_support_data(data_1, support_data_index):
    saved_data = list()
    for i in range(5):
        # print(i)
        temp = data_1[i][support_data_index]
        saved_data.append(temp)
    for i in range(5,10):
        saved_data.append(data_1[i])
    return saved_data

def augmentation(data, rate):
    out = list()
    for i in range(len(data)):
        out.append(np.repeat(data[i], rate, axis=0))
    return out

def get_class_average(feature, label):
    unique_label = sorted(list(set(label)))
    class_average = list()
    feature = np.array(feature)
    for i in unique_label:
        related_feature = feature[label==i]
        average = np.mean(related_feature, axis=0)
        class_average.append(average)
    return np.array(class_average)

def construct_examplar(final_train_fea, label, total_size, all_label):
    final_train_fea = np.array(final_train_fea)
    sample_ratio = float(total_size)/float(len(label))
    from scipy.spatial import distance
    label_set = set(label)
    final_index = list()
    for l in label_set:
        # print(l)
        index_specific_label = np.where(label==l)[0]
        fea_specific_label = final_train_fea[index_specific_label]
        average = np.mean(fea_specific_label, axis=0)
        num_select = int(len(index_specific_label)*sample_ratio)
        distance_center = distance.cdist(fea_specific_label, 
            np.expand_dims(average, axis=0))
        distance_center = distance_center.flatten()
        selected_index = np.argsort(distance_center)[:num_select]
        original_index = index_specific_label[selected_index]
        final_index += list(original_index)
    return final_index    
