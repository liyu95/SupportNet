import numpy as np
import random
import copy
from sklearn.cross_validation import train_test_split
#non_enzyme:0, EC.1:1 and so on
#This function would help to construct the sequence and label dictionary
def add_into_class_dict(sequence_list, start, end, class_num, dictionary):
	for i in range(start,end):
		dictionary[sequence_list[i][1:]]=class_num
	return dictionary


#This function would constuct the sequence and label dictionary
#input: sequence list, class_notation_index: the index which indicate the position
#which separate different classes, those positions a not a protein sequence, but still
#elements of the list
#output: a dictionary, each key is a protein sequence in string, each value
#is a class
def construct_label_dictionary(sequence_list, class_notation_index):
	dictionary={}
	dictionary=add_into_class_dict(sequence_list, class_notation_index[-1], len(sequence_list), 0, dictionary)
	for i in range(len(class_notation_index)-1):
		dictionary=add_into_class_dict(sequence_list, class_notation_index[i], class_notation_index[i+1], i+1, dictionary)
	return dictionary


#This function would construct the sequence and feature dictionary
#input: sequence list, which contain some useless sequence to indicate the
#separation of different classes. class_notation_index: the index of those
#useless positions, which we would ignore. feature_list: corresponding to the 
#pure sequence list, has the same order
#output: a dictionary, each key is a protein sequence in string, each value
#is a the corresponding feature
def construct_feature_dictionary(sequence_list,class_notation_index,feature_list):
	feature_dictionary={}
	sequence_list_copy=copy.copy(sequence_list)
	for i in range(len(class_notation_index)-1,-1,-1):
		sequence_list_copy.pop(class_notation_index[i])
	print('The length of sequence list is %d'%len(sequence_list))
	print('The length of sequence list copy is %d'%len(sequence_list_copy))
	for i in range(len(sequence_list_copy)):
		feature_dictionary[sequence_list_copy[i][1:]]=feature_list[i]
	return feature_dictionary


#this function would generate train and test sequence according to the ratio of different classes
#input: sequence list. non_enzyme_index: indicate the position which separate different classes.
#train_ratio, random_seed
#output: two list, train_data, test_data
#This function generate the data according to the ratio of different classes, to avoid 0-sample class
def train_test_data_generation_level_0_and_1(sequence_list,non_enzyme_index,train_ratio,random_seed,level=0):
	train_data=[]
	test_data=[]
	if level==0:
		train_data,test_data=train_test_split(sequence_list[non_enzyme_index[-1]+1:len(sequence_list)], 
            train_size=train_ratio, random_state=random_seed)
	for i in range(len(non_enzyme_index)-1):
		train_data_temp,test_data_temp=train_test_split(sequence_list[non_enzyme_index[i]+1:non_enzyme_index[i+1]], 
            train_size=train_ratio, random_state=random_seed)
		train_data=train_data+train_data_temp
		test_data=test_data+test_data_temp
	random.seed(random_seed)
	random.shuffle(train_data)
	random.shuffle(test_data)
	for i in range(len(train_data)):
		train_data[i]=train_data[i][1:]
	for i in range(len(test_data)):
		test_data[i]=test_data[i][1:]
	return (train_data,test_data)

#This function would generate the label array corresponding to the sequence list and the label_dictioanry.
#If level is equal to 0, enzyme is 1, non-enzyme is 0
#If level is equal to 1, we have 6 classes.
#If level is equal to 2, we have 49 classes.
def label_array_generation(sequence_list,label_dictionary,level):
	label_list=[]
	if level==0:
		for i in range(len(sequence_list)):
			if label_dictionary[sequence_list[i]]==0:
				label_list.append(0)
			else:
				label_list.append(1)
	if level==1 or level==2 or level==3:
		for i in range(len(sequence_list)):
			label_list.append(label_dictionary[sequence_list[i]])
	label_array=np.array(label_list)
	return label_array


#This function would extend the original feature (sequence_length*20) to (target_lenght*20), complete with 0s.
#Please do target_length which is larger than sequence length
def feature_length_extend(original_feature,target_length):
	if target_length-np.shape(original_feature)[0]<0:
		print("You target length is smaller than the original length!!")
	complementry_array=np.zeros((target_length-np.shape(original_feature)[0],np.shape(original_feature)[1]))
	return np.vstack((original_feature,complementry_array))


#This function would generate the pssm feature array corresponding to the sequence list, according to the label dictioanry
#If target_length is larger than 0, we would extend the feature to the target length.
def feature_array_generation(sequence_list,feature_dictionary_original_length,target_length=0):
	feature_list=[]
	for i in range(len(sequence_list)):
		feature_list.append(feature_dictionary_original_length[sequence_list[i]])
	if target_length>0:
		for i in range(len(feature_list)):
			feature_list[i]=feature_length_extend(feature_list[i],target_length)
			if i%1000==0:
				print("Process %dth sequence feature."%i)
	feature_array=np.array(feature_list)
	return feature_array

#This function owuld encode a single sequence into a 2 dimension list
def single_sequence_encoding(sequence):
    sequence_matrix=[]
    AA_list=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    for single_letter in sequence:
        encoding_temp=np.zeros((20))
        try:
            position_index=AA_list.index(single_letter)
        except ValueError:
            print(sequence)
        else:
            encoding_temp[position_index]=1
        sequence_matrix.append(encoding_temp)
    return sequence_matrix


#This function would encoding the protein sequence into a (sequence_length*20) vector.
#Input: sequence_list: a list of sequence we would like to encode.
# 		target_length: the length we would like to extend the sequence encoding to, if the target_length is 0, we would not do the extension.
#Output: a list, each element is a 2 dimension numpy array
def protein_sequence_encoding(sequence_list,target_length=0):
	encoded_list=[]
	for sequence in sequence_list:
		single_sequence_encoding_result=single_sequence_encoding(sequence)
		encoded_list.append(single_sequence_encoding_result)
	if target_length>0:
		for i in range(len(encoded_list)):
			encoded_list[i]=feature_length_extend(encoded_list[i],target_length)
			if i%1000==0:
				print("Process %dth sequence feature."%i)
	return encoded_list

#function to generate feed batch
def generate_feeding_batch(train_feature,train_label,batch_size):
    from numpy.random import randint
    batch_index=randint(0,len(train_feature),batch_size)
    feature_batch=[]
    label_batch=[]
    for index in batch_index:
        feature_batch.append(train_feature[index])
        label_batch.append(train_label[index])
    feature_batch=np.array(feature_batch)
    label_batch=np.array(label_batch)
    feature_batch=feature_batch.astype('float')
    label_batch=label_batch.astype('float')
    return (feature_batch,label_batch)

def to_categorical(label_array,total_classes):
    from sklearn.preprocessing import OneHotEncoder
    enc=OneHotEncoder()
    label_list=[]
    for i in range(len(label_array)):
        label_list.append([label_array[i]])
    return enc.fit_transform(label_list).toarray()


def label_one_hot(train_label,test_label,level,hiera=False):
    if hiera==False:
        if level==0:
            train_label_categorical=to_categorical(train_label,2)
            test_label_categorical=to_categorical(test_label,2)
        if level==1:
            train_label_categorical=to_categorical(train_label-1,6)
            test_label_categorical=to_categorical(test_label-1,6)
        if level==2:
            train_label_categorical=to_categorical(train_label-1,49)
            test_label_categorical=to_categorical(test_label-1,49)
    if hiera==True:
        if level==1:
            train_label_categorical=to_categorical(train_label,7)
            test_label_categorical=to_categorical(test_label,7)
        if level==2:
            train_label_categorical=to_categorical(train_label,50)
            test_label_categorical=to_categorical(test_label,50)
    return train_label_categorical,test_label_categorical

def pre_pssm_single(pssm_matrix,maximum_dimen):
#     input: a single profile: sequence_length*20;maximum_dimen: the furthest correlation between two aa, it should be small than the shortest length
#     output: a numpy vector: 20+20*maximum_dimen
    pssm_mean=np.mean(pssm_matrix,axis=0)
    correlation=np.zeros([maximum_dimen,20])
    for i in range(maximum_dimen):
        for j in range(20):
            temp=0.0
            for k in range(np.size(pssm_matrix,axis=0)-(i+1)):
                temp=temp+(pssm_matrix[k,j]-pssm_matrix[k+(i+1),j])**2
            temp=temp/float(np.size(pssm_matrix,axis=0)-(i+1))
            correlation[i,j]=temp
    correlation_reshape=np.reshape(correlation,np.size(correlation,axis=0)*np.size(correlation,axis=1))
    return np.hstack((pssm_mean,correlation_reshape))

def pre_pssm_list(pssm_list,maximum_dimen):
#   given the input and the maximum_dimen, get the pre_pssm
    pre_pssm=[]
    for i in range(len(pssm_list)):
        pre_pssm.append(pre_pssm_single(pssm_list[i],maximum_dimen))
        if i%1000==0:
            print('Processing %dth pssm'%i)
    pre_pssm_array=np.array(pre_pssm)
    return pre_pssm_array

def get_paac_sequence(sequence,lamb):
    from pydpi.pypro import PyPro
    protein=PyPro()
    protein.ReadProteinSequence(sequence)
    dictionary=protein.GetPAAC(lamda=lamb)
    result=[]
    for i in range(1,20+lamb+1):
        result.append(dictionary['PAAC'+str(i)])
    return result

#The input could contain the class notation index and '>'
def get_paac_list(sequence_list,class_notation_index=[],lamb=30):
    import copy
    from pydpi.pypro import PyPro
    sequence_list_temp=copy.copy(sequence_list)
    for i in range(len(class_notation_index)-1,-1,-1):
        sequence_list_temp.pop(class_notation_index[i])
    print('length of sequence list temp: %d'%len(sequence_list_temp))
    result=[]
    i=0
    for sequence in sequence_list_temp:
        i=i+1
        if i%1==0:
            print('Processing %dth sequence'%i)
        try:
            result.append(get_paac_sequence(sequence[1:],lamb))
        except AttributeError:
            print('The %dth sequence has more than 20 types aa'%i)
            print(sequence)
            sequence_temp=delete_unusual_aa(sequence)
            print(sequence_temp)
            result.append(get_paac_sequence(sequence_temp,lamb))
    return result

# This function would help you to find the unusual sequence in a sequence list
def find_unusual_sequence(sequence_list):
    import copy
    from pydpi.pypro import PyPro
    sequence_list_temp=copy.copy(sequence_list)
    unusual_sequence_index=[]
    for i in range(len(sequence_list_temp)):
        try:
            protein=PyPro()
            protein.ReadProteinSequence(sequence_list[i][1:])
            dictionary=protein.GetAAComp()
        except AttributeError:
            print('The %dth sequence has more than 20 types aa'%i)
            unusual_sequence_index.append(i)
    return unusual_sequence_index

# This function would help to delete the unusual aa from the sequence
def delete_unusual_aa(sequence):
    import copy
    AA_list=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    sequence_temp=copy.copy(sequence)
    del_aa_index=[]
    for i in range(len(sequence_temp)):
        if sequence_temp[i] not in AA_list:
            del_aa_index.append(i)
    for i in range(len(del_aa_index)-1,-1,-1):
        sequence_temp=sequence_temp[:del_aa_index[i]]+sequence_temp[del_aa_index[i]+1:]
    return sequence_temp

def get_distance_fre(sequence,option):
    basic_AA=['H','K','R']
    hydrophobic_AA=['I','V','L','F','M','A','G','W','P']
    other_AA=['D','N','E','Q','Y','S','T','C']
    index=[]
    if option=='basic':
        for i in range(len(sequence)):
            if sequence[i] in basic_AA:
                index.append(i)
    if option=='hydrophobic':
        for i in range(len(sequence)):
            if sequence[i] in hydrophobic_AA:
                index.append(i)
    if option=='other':
        for i in range(len(sequence)):
            if sequence[i] in other_AA:
                index.append(i)
    distance=[]
    for i in range(len(index)-1):
        distance.append(index[i+1]-index[i])
    frequency=np.zeros(6)
    for i in distance:
        if i==1:
            frequency[0]+=1
        if 1<i<=6:
            frequency[1]+=1
        if 6<i<=11:
            frequency[2]+=1
        if 11<i<=16:
            frequency[3]+=1
        if 16<i<=21:
            frequency[4]+=1
        if i>21:
            frequency[5]+=1
    if sum(frequency)>0:
        frequency=frequency/float(sum(frequency))
    return list(frequency)

def get_twin_frequency(sequence):
    frequence=np.zeros(20)
    AA_list=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    for i in range(len(sequence)-1):
        if sequence[i]==sequence[i+1]:
            index_temp=AA_list.index(sequence[i])
            frequence[index_temp]+=1
    if sum(frequence)>0:
        frequence=frequence/float(sum(frequence))
    return list(frequence)

def get_aa_frequency(sequence):
    from pydpi.pypro import PyPro
    protein=PyPro()
    protein.ReadProteinSequence(sequence)
    dictionary=protein.GetAAComp()
    AA_list=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    result=[]
    for aa in AA_list:
        temp=dictionary[aa]/float(100)
        result.append(temp)
    return result

# dN=22,dC=9
def three_parts_representation_sequence(sequence):
    from pydpi.pypro import PyPro
    dN=22
    dC=9
    representation=[]
    if len(sequence)>=(4*dN+20+dC):
        n1=sequence[:dN]
        n2=sequence[dN:2*dN]
        n3=sequence[2*dN:3*dN]
        n4=sequence[3*dN:4*dN]
        n=sequence[:4*dN]
        m=sequence[4*dN:-dC]
        c=sequence[-dC:]
    elif len(sequence)>(4*dN+dC):
        n1=sequence[:dN]
        n2=sequence[dN:2*dN]
        n3=sequence[2*dN:3*dN]
        n4=sequence[3*dN:4*dN]
        n=sequence[:4*dN]
        m=sequence[-20-dC:-dC]
        c=sequence[-dC:]
    else:
        n1=sequence[:(len(sequence)-dC)/2]
        n2=sequence[:(len(sequence)-dC)/2]
        n3=sequence[:(len(sequence)-dC)/2]
        n4=sequence[:(len(sequence)-dC)/2]
        n=sequence[:(len(sequence)-dC)/2]
        m=sequence[(len(sequence)-dC)/2:-dC]
        c=sequence[-dC:]
    temp=get_aa_frequency(n1)
    representation=representation+temp
    temp=get_aa_frequency(n2)
    representation=representation+temp
    temp=get_aa_frequency(n3)
    representation=representation+temp
    temp=get_aa_frequency(n4)
    representation=representation+temp
    temp=get_aa_frequency(m)
    representation=representation+temp
    temp=get_twin_frequency(m)
    representation=representation+temp
    temp=get_aa_frequency(c)
    representation=representation+temp
    temp=get_distance_fre(n,'basic')
    representation=representation+temp
    temp=get_distance_fre(m,'basic')
    representation=representation+temp
    temp=get_distance_fre(m,'hydrophobic')
    representation=representation+temp
    temp=get_distance_fre(m,'other')
    representation=representation+temp
    temp=get_aa_frequency(sequence)
    representation=representation+temp
    if max(representation)!=0:
        representation=np.array(representation)/float(max(representation))
    return list(representation)

# This funciton is based on the Japanese paper, which would use 184 component to represent a protein sequence
def three_parts_representation_list(sequence_list,class_notation_index=[]):
    import copy
    from pydpi.pypro import PyPro
    sequence_list_temp=copy.copy(sequence_list)
    for i in range(len(class_notation_index)-1,-1,-1):
        sequence_list_temp.pop(class_notation_index[i])
    print('length of sequence list temp: %d'%len(sequence_list_temp))
    result=[]
    for sequence in sequence_list_temp:
        try:
            protein=PyPro()
            protein.ReadProteinSequence(sequence[1:])
            dictionary=protein.GetAAComp()
        except AttributeError:
            sequence=delete_unusual_aa(sequence)
            sequence='>'+sequence
        temp=three_parts_representation_sequence(sequence[1:])
        result.append(temp)
    return result

def Calculate_CTF(sequence_list):
    import numpy as np
    from pydpi import pypro
    feature=[]
    for sequence in sequence_list:
        dic_temp=pypro.CalculateConjointTriad(sequence)
        dic_temp=sorted(dic_temp.iteritems(), key=lambda d:d[0])
        for i in xrange(len(dic_temp)-1,-1,-1):
            if '0' in dic_temp[i][0]:
                dic_temp.pop(i)
        feature_temp=[]
        for tup in dic_temp:
            feature_temp.append(tup[1])
        feature.append(feature_temp)
    return np.array(feature)

# The following 4 functions are used for delete sequences which is too small
def find_del_label(statistic, threshold):
    del_list = []
    for i in range(len(statistic)-1, -1, -1):
        if statistic[i][1] < threshold:
            del_list.append(statistic[i][0])
    return del_list

def find_del_ind(label_list, del_label_list):
    del_index = []
    for i in range(len(label_list)):
        if label_list[i] in del_label_list:
            del_index.append(i)
    return del_index
def pop_out(sequence_list, del_index):
    for i in range(len(del_index)-1, -1, -1):
        sequence_list.pop(del_index[i])
    return sequence_list

def delete_small_classes(sequence_list, label_list, threshold):
    from collections import Counter
    statistic = Counter(label_list).most_common()
    del_label_list = find_del_label(statistic, threshold)
    del_index = find_del_ind(label_list, del_label_list)
    print del_label_list
    print del_index
    sequence_list = pop_out(sequence_list, del_index)
    return sequence_list

if __name__ == '__main__':
	pass