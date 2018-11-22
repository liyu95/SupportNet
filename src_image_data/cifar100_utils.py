import pandas as pd
import numpy as np
from collections import defaultdict
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
import pickle
CIFAR100_TRAIN_FILE='cifar-100-python/train'
CIFAR100_TEST_FILE='cifar-100-python/test'
CIFAR100_META_FILE='cifar-100-python/meta'
CIFAR10_TRAIN_FILE_PREFIX='cifar-10-batches-py/data_batch'
CIFAR10_TEST_FILE='cifar-10-batches-py/test_batch'
CIFAR10_META_FILE='cifar-10-batches-py/batches.meta'
def coarse_label_to_names(coarse_label_list):
    meta=unpickle(CIFAR100_META_FILE,encoding='latin-1')
    return [meta['coarse_label_names'][i] for i in coarse_label_list]
def fine_label_to_names(fine_label_list):
    meta=unpickle(CIFAR100_META_FILE,encoding='latin-1')
    return [meta['fine_label_names'][i] for i in fine_label_list]
def unpickle(file,encoding):
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding=encoding)
    fo.close()
    return dict
def load_cifar100_data(samples_per_class_on_validation=0,normalize=True):
    xs = []
    ys = []
    for j in range(1):
        d = unpickle(CIFAR100_TRAIN_FILE,encoding='latin-1')
        x = d['data']
        y = d['fine_labels']
        xs.append(x)
        ys.append(y)

    d = unpickle(CIFAR100_TEST_FILE,encoding='latin-1')
    xs.append(d['data'])
    ys.append(d['fine_labels'])
    if normalize:
        x = np.concatenate(xs)/np.float32(255)
    else:
        x = np.concatenate(xs)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))
    # subtract per-pixel mean
    if normalize:
        pixel_mean = np.mean(x[0:50000],axis=0)
         # x -= pixel_mean
    # Create Train/Validation set
    eff_samples_cl = 500-samples_per_class_on_validation
    X_train = np.zeros((eff_samples_cl*100,32,32,3))
    Y_train = np.zeros(eff_samples_cl*100)
    X_valid = np.zeros((samples_per_class_on_validation*100,32,32,3))
    Y_valid = np.zeros(samples_per_class_on_validation*100)
    for i in range(100):
        index_y=np.where(y[0:50000]==i)[0]
        np.random.shuffle(index_y)
        X_train[i*eff_samples_cl:(i+1)*eff_samples_cl] = x[index_y[0:eff_samples_cl],:,:,:]
        Y_train[i*eff_samples_cl:(i+1)*eff_samples_cl] = y[index_y[0:eff_samples_cl]]
        X_valid[i*samples_per_class_on_validation:(i+1)*samples_per_class_on_validation] = x[index_y[eff_samples_cl:500],:,:,:]
        Y_valid[i*samples_per_class_on_validation:(i+1)*samples_per_class_on_validation] = y[index_y[eff_samples_cl:500]]

    X_test  = x[50000:,:,:,:]
    Y_test  = y[50000:]
    if normalize:
        return dict(
            X_train = X_train.astype('float32'),
            Y_train = Y_train.astype('int32'),
            X_valid = X_valid.astype('float32'),
            Y_valid = Y_valid.astype('int32'),
            X_test  = X_test.astype('float32'),
            Y_test  = Y_test.astype('int32'))
    else:
        return dict(
            X_train = X_train.astype('uint8'),
            Y_train = Y_train.astype('int32'),
            X_valid = X_valid.astype('uint8'),
            Y_valid = Y_valid.astype('int32'),
            X_test  = X_test.astype('uint8'),
            Y_test  = Y_test.astype('int32'))
def load_cifar10_data(samples_per_class_on_validation=0):
    xs = []
    ys = []
    for j in range(5):
        d = unpickle(CIFAR10_TRAIN_FILE_PREFIX+'_'+str(j+1),encoding='bytes')
        x = d[b'data']
        y = d[b'labels']
        xs.append(x)
        ys.append(y)

    d = unpickle(CIFAR10_TEST_FILE,encoding='bytes')
    xs.append(d[b'data'])
    ys.append(d[b'labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))
    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    #x -= pixel_mean
    # Create Train/Validation set
    eff_samples_cl = 5000-samples_per_class_on_validation
    X_train = np.zeros((eff_samples_cl*10,32,32,3))
    Y_train = np.zeros(eff_samples_cl*10)
    X_valid = np.zeros((samples_per_class_on_validation*10,32,32,3))
    Y_valid = np.zeros(samples_per_class_on_validation*10)
    for i in range(10):
        index_y=np.where(y[0:50000]==i)[0]
        np.random.shuffle(index_y)
        X_train[i*eff_samples_cl:(i+1)*eff_samples_cl] = x[index_y[0:eff_samples_cl],:,:,:]
        Y_train[i*eff_samples_cl:(i+1)*eff_samples_cl] = y[index_y[0:eff_samples_cl]]
        X_valid[i*samples_per_class_on_validation:(i+1)*samples_per_class_on_validation] = x[index_y[eff_samples_cl:5000],:,:,:]
        Y_valid[i*samples_per_class_on_validation:(i+1)*samples_per_class_on_validation] = y[index_y[eff_samples_cl:5000]]

    X_test  = x[50000:,:,:,:]
    Y_test  = y[50000:]

    return dict(
        X_train = X_train.astype('float32'),
        Y_train = Y_train.astype('int32'),
        X_valid = X_valid.astype('float32'),
        Y_valid = Y_valid.astype('int32'),
        X_test  = X_test.astype('float32'),
        Y_test  = Y_test.astype('int32'))
