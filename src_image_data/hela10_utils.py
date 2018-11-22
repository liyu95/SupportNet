import os.path
import numpy as np
from scipy.misc import imresize,imread
import pandas as pd
import numpy as np
from collections import defaultdict,Counter
from itertools import combinations
from itertools import chain
from collections import namedtuple
import pickle
import tensorflow as tf
import tflearn
import matplotlib.pyplot as plt
import os.path
import shutil
import inspect
from nn_lib import *
import cifar100_utils
import matplotlib.pyplot as plt
import json
from pprint import pprint
import train_utils
from scipy.spatial.distance import cdist
import time
from glob import glob
from train_utils import *
np.random.seed(1997)
hela10_meta=dict(label_names=\
                ['ActinFilaments',
                 'ER',
                 'Endosome',
                 'Golgi_gia',
                 'Golgi_gpp',
                 'Lysosome',
                 'Microtubules',
                 'Mitochondria',
                 'Nucleolus',
                 'Nucleus'])
def load_hela10_data(number_of_test=200):
    return assemble_dataset('./hela10',number_of_test)
def assemble_dataset(base_dir,number_of_test=200):
    X=[]
    Y=[]
    original_file_names=[]
    length=0
    for cl,cl_name in enumerate(hela10_meta['label_names']):
        file_names=glob(os.path.join(base_dir,cl_name,'*.png'))
        for f in file_names:
            img=imread(f)
            if np.all(img.shape==np.array([382,512])):
                img=img.T
            if not np.all(img.shape==np.array([512,382])):
                print(img.shape)
                print(f)
            assert np.all(img.shape==np.array([512,382]))
            X.append(img)
            Y.append(cl)
            original_file_names.append(f)
    X=np.stack(X)
    Y=np.array(Y)
    original_file_names=np.array(original_file_names)
    #fix class imbalance
    indices={}
    c=Counter(Y)
    for i in c.keys():
        indices[i]=np.where(Y==i)[0]
    max_class=max(c.values())

    for i in indices:
        if len(indices[i])<max_class:
            rep_num=int(max_class/len(indices[i]))
            remainder=max_class%len(indices[i])
            indices[i]=np.tile(indices[i],rep_num)
            indices[i]=np.concatenate([indices[i],np.random.choice(indices[i],remainder)],axis=0)
    indices=np.concatenate(list(indices.values()))
    cc=Counter(Y[indices])
    lens=list(cc.values())
    assert np.all(np.array(lens)==lens[0])
    X=X[indices,...]
    Y=Y[indices]
    original_file_names=original_file_names[indices]    
    
    #train_test split
    num_test_per_label=int(number_of_test/len(hela10_meta['label_names']))
    train_indices=[]
    test_indices=[]
    for i in range(len(hela10_meta['label_names'])):
        selected_index=np.where(Y==i)[0]
        assert len(selected_index)>=num_test_per_label
        np.random.shuffle(selected_index)
        train_indices.append(selected_index[num_test_per_label:])
        test_indices.append(selected_index[:num_test_per_label])
    train_indices=np.concatenate(train_indices)
    test_indices=np.concatenate(test_indices)
    return dict(X_train=X[train_indices,:,:,np.newaxis],
                Y_train=Y[train_indices],
                original_file_names_train=original_file_names[train_indices],
                X_test=X[test_indices,:,:,np.newaxis],
                Y_test=Y[test_indices],
                original_file_names_test=original_file_names[test_indices])
                        
