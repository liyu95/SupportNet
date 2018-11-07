import numpy as np
import tflearn
from sklearn.model_selection import train_test_split
import random
import cPickle
import copy
import math
import sys
sys.path.append('/home/liy0f/ec_project/data_and_feature')
import protein_sequence_process_functions as p_func
import Pfam_pickle_file_to_array_encoding as Pfam
import tensorflow as tf
import os


#Global variable, MAX_LENGTH is the maximum length of all sequence.
TEST=False
DROPOUT=False
MAX_LENGTH=5000
TYPE_OF_AA=20
DOMAIN=16306
LOAD=False
train_ratio=0.9
level=2
n_class=1
# data_size need to be determined when get the sequence belonging to a certain class
data_size=0

def label_one_hot(train_label,test_label,number_classes):
    train_label_categorical=p_func.to_categorical(train_label,number_classes)
    test_label_categorical=p_func.to_categorical(test_label,number_classes)
    return train_label_categorical,test_label_categorical

def construct_feature_dictionary(sequence_list,feature_list):
    feature_dictionary={}
    print('The length of sequence list is %d'%len(sequence_list))
    for i in range(len(sequence_list)):
        feature_dictionary[sequence_list[i].split('>')[1]]=feature_list[i]
    return feature_dictionary

def construct_label_dictionary(sequence_list,level):
    label_dictionary={}
    if level<=1:
        for i in range(len(sequence_list)):
            label_dictionary[sequence_list[i].split('>')[1]]=int(sequence_list[i].split('.')[0][1:])
    if level==2:
        for i in range(len(sequence_list)):
            label_dictionary[sequence_list[i].split('>')[1]]=int(sequence_list[i].split('.')[1])
    return label_dictionary
def train_test_data_generation(sequence_list,label_list,train_ratio,random_seed):
    train_data=[]
    test_data=[]
    unique_label=list(set(label_list))
    index_list=[]
    for i in range(len(unique_label)):
        index_list.append(label_list.index(unique_label[i]))
    index_list.sort()
    for i in range(len(unique_label)-1):
        train_data_temp,test_data_temp=train_test_split(sequence_list[index_list[i]:index_list[i+1]], 
            train_size=train_ratio, random_state=random_seed)
        train_data=train_data+train_data_temp
        test_data=test_data+test_data_temp
    train_data_temp,test_data_temp=train_test_split(sequence_list[index_list[-1]:len(sequence_list)], 
        train_size=train_ratio, random_state=random_seed)
    train_data=train_data+train_data_temp
    test_data=test_data+test_data_temp
    random.seed(random_seed)
    random.shuffle(train_data)
    random.shuffle(test_data)
    for i in range(len(train_data)):
        train_data[i]=train_data[i].split('>')[1]
    for i in range(len(test_data)):
        test_data[i]=test_data[i].split('>')[1]
    return (train_data,test_data)

def encoding_to_1d(sequence_encoding_array):
    encoding_1d_list=[]
    for i in range(len(sequence_encoding_array)):
        temp_array=np.nonzero(sequence_encoding_array[i])[1]+1
        np.transpose(temp_array)
        temp_array=np.append(temp_array,np.zeros(MAX_LENGTH-len(temp_array)))
        encoding_1d_list.append(temp_array)
    return np.array(encoding_1d_list)

#load sequence data
def load_level_2_data(level, n_class):

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

	#filter those sequence that belong to the certain class in level 1
	def filter_sequence(sequence_list):
	    se_list_filtered=[]
	    for i in range(len(sequence_list)):
	        if int(sequence_list[i].split('.')[0][1:])==n_class:
	            se_list_filtered.append(sequence_list[i])
	    return se_list_filtered

	sequence_list=filter_sequence(sequence_list)

	#The number of sequence that belong to a certain subclass is too small
	#We should eliminate those classes by deleting those sequences
	if n_class==1:
	    sequence_list.pop(2027)
	if n_class==3:
	    sequence_list.pop(5916)
	    sequence_list.pop(2088)
	    sequence_list.pop(2087)
	    sequence_list.pop(2079)
	if n_class==4:
	    sequence_list.pop(1483)
	    sequence_list.pop(1370)

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

	#mapping the label to continuous label space so that easier for one hot
	unique_label=list(set(label_list))
	unique_label.sort()
	label_mapping_dict={}
	for i in range(len(unique_label)):
	    label_mapping_dict[unique_label[i]]=i
	train_label_temp=[]
	test_label_temp=[]
	for i in range(len(train_label)):
	    train_label_temp.append(label_mapping_dict[train_label[i]])
	for i in range(len(test_label)):
	    test_label_temp.append(label_mapping_dict[test_label[i]])
	train_label_temp=np.array(train_label_temp)
	test_label_temp=np.array(test_label_temp)

	train_label=train_label_temp
	test_label=test_label_temp

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

# class number start from 0
def exclude_data(data, e_class_list):
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

	num_labels = len(set(train_label))

	train_label_categorical = train_label_categorical[np.isin(train_label, e_class_list, invert=True)]
	train_pssm_full_length = train_pssm_full_length[np.isin(train_label, e_class_list, invert=True)] 
	train_encoding_full_length = train_encoding_full_length[np.isin(train_label, e_class_list, invert=True)]
	train_functional_domain_encoding = train_functional_domain_encoding[np.isin(train_label, e_class_list, invert=True)]
	train_label = train_label[np.isin(train_label, e_class_list, invert=True)]

	test_label_categorical = test_label_categorical[np.isin(test_label, e_class_list, invert=True)]
	test_pssm_full_length = test_pssm_full_length[np.isin(test_label, e_class_list, invert=True)]
	test_encoding_full_length = test_encoding_full_length[np.isin(test_label, e_class_list, invert=True)]
	test_functional_domain_encoding =  test_functional_domain_encoding[np.isin(test_label, e_class_list, invert=True)]
	test_label = test_label[np.isin(test_label, e_class_list, invert=True)]

	# train_label_categorical,test_label_categorical=label_one_hot(train_label,test_label,num_labels)

	return (train_label, train_label_categorical, train_pssm_full_length, train_encoding_full_length, 
		train_functional_domain_encoding, test_label, test_label_categorical, test_pssm_full_length,
		test_encoding_full_length, test_functional_domain_encoding)

