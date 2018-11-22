import pandas as pd
import numpy as np
from itertools import combinations
from itertools import chain
from collections import namedtuple,defaultdict,Counter
from scipy.misc import imresize,imread
import pickle
import tensorflow as tf
import tflearn
import matplotlib.pyplot as plt
import os.path
import shutil
import inspect
from copy import copy
from nn_lib import *
from cifar100_utils import *
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import cdist
from pprint import pprint
from sklearn.svm import SVC
np.random.seed(1997)
def all_done():
    from IPython.display import Audio, display
    display(Audio(url='https://sound.peal.io/ps/audios/000/000/537/original/woo_vu_luvub_dub_dub.wav', autoplay=True))

def assert_session(session):
    if session is None:
        session=tf.get_default_session()
    assert session is not None
    return session

def convert_numpy_dtype(d):
    for k,v in d.items():
        if type(v) in [np.int32,np.int64]:
            d[k]=int(v)
        elif type(v) in [np.float32,np.float64]:
            d[k]=float(v)

def print_eval_metric(log_vars,print_var_list,print_var_name_list,title):
    print_dict={v:log_vars[v] for v in print_var_list}
    print_var_name_dict=dict(zip(print_var_list,print_var_name_list))
    print(title)
    for v in print_var_list:
        print('\t%s: %f'%(print_var_name_dict[v],log_vars[v]))
def reset_log_vars(tf_log_var_val,reset_var_list):
    for v in reset_var_list:
        if v in ['local_step','global_step','iteration_step',
                 'top1_accuracy_train','top1_accuracy_test','best_top1_accuracy_test',
                 'top5_accuracy_train','top5_accuracy_test','best_top5_accuracy_test']:
            tf_log_var_val[v]=0
        elif v in ['loss_train','loss_test','best_loss_test']:
            tf_log_var_val[v]=np.Infinity
        elif v in ['best_loss_test_epoch']:
            tf_log_var_val[v]=0
        elif v in ['best_top1_accuracy_test_epoch','best_top5_accuracy_test_epoch']:
            tf_log_var_val[v]=0
def add_to_dict(log_vars,evaluation_vals,suffix,var_list):
    for k in var_list:
        if suffix:
            k_suffix=k+'_%s'%(suffix)
        else:
            k_suffix=k
        v=evaluation_vals[k]
        log_vars[k_suffix]=v
    return log_vars

def add_to_list(history,evaluation_vals,suffix,var_list):

    for k in var_list:
        if suffix:
            k_suffix=k+'_%s'%(suffix)
        else:
            k_suffix=k
        v=evaluation_vals[k]
        if k_suffix in history:
            history[k_suffix].append(v)
        else:
            history[k_suffix] = [v]
    return history

def test_accuracy_evaluation_plain(X_test,Y_test,test_batch_size,evaluation_tensors,session=None,dataset='cifar100'):
    if session is None:
        session=tf.get_default_session()
    assert session is not None
    evaluation_vals={}
    evaluation_vals['loss']=0
    evaluation_vals['class_loss']=0
    evaluation_vals['regularization_loss']=0
    evaluation_vals['top1_accuracy']=0
    evaluation_vals['top5_accuracy']=0
    Y_pred=[]
    for X_minibatch,Y_minibatch in iterate_minibatches(X_test,Y_test,test_batch_size,shuffle=False,augment=False,dataset=dataset,yield_remaining=True,fix_imbalance=False):
        local_loss,local_class_loss,local_regularization_loss,local_top1_accuracy_test,local_top5_accuracy_test,pred=session.run([evaluation_tensors['loss'],evaluation_tensors['class_loss'],evaluation_tensors['regularization_loss'],evaluation_tensors['top1_accuracy'],evaluation_tensors['top5_accuracy'],evaluation_tensors['fc']],feed_dict={evaluation_tensors['X']:X_minibatch,
                               evaluation_tensors['Y']:Y_minibatch})
        weight=len(X_minibatch)/len(X_test)
        evaluation_vals['loss']+=local_loss*weight
        evaluation_vals['class_loss']+=local_class_loss*weight
        evaluation_vals['regularization_loss']+=local_regularization_loss*weight
        evaluation_vals['top1_accuracy']+=local_top1_accuracy_test*weight
        evaluation_vals['top5_accuracy']+=local_top5_accuracy_test*weight
        Y_pred.append(pred.argmax(axis=1))
    Y_pred=np.concatenate(Y_pred)
    sklearn_eval_vals=sklearn_evaluation(Y_test,Y_pred)
    assert np.abs(sklearn_eval_vals['top1_accuracy']-evaluation_vals['top1_accuracy'])<1e-2
    evaluation_vals.update(sklearn_eval_vals)
    evaluation_vals['top1_accuracy']=float(evaluation_vals['top1_accuracy'])
    return evaluation_vals
def test_prediction_plain(X_test,test_batch_size,evaluation_tensors,session=None,dataset='cifar100'):
    if session is None:
        session=tf.get_default_session()
    assert session is not None
    predictions=[]
    #dummy Y_test
    Y_test=np.zeros_like(X_test)
    for X_minibatch,Y_minibatch in iterate_minibatches(X_test,Y_test,test_batch_size,shuffle=False,augment=False,yield_remaining=True,dataset=dataset,fix_imbalance=False):
        pred,=\
        session.run([evaluation_tensors['fc']],
                    feed_dict={evaluation_tensors['X']:X_minibatch})
        predictions.append(pred.argmax(axis=1))
    return np.concatenate(predictions)
def iterate_minibatches(inputs, targets, batchsize, dataset='cifar100',shuffle=False, augment=False,yield_remaining=False,
                        image_normalization=True,fix_imbalance=False):
    data_augmentation_rate=2
    if dataset=='cifar10':
        return iterate_minibatches_cifar100(inputs=inputs, targets=targets, batchsize=batchsize,
                                            shuffle=shuffle,augment=augment,
                                            yield_remaining=yield_remaining)
    if dataset=='cifar100':
        return iterate_minibatches_cifar100(inputs=inputs, targets=targets, batchsize=batchsize,
                                            shuffle=shuffle,augment=augment,
                                            yield_remaining=yield_remaining)

    elif dataset=='breakhis':
        return iterate_minibatches_breakhis(inputs=inputs, targets=targets, batchsize=batchsize,
                                            shuffle=shuffle, augment=augment,yield_remaining=yield_remaining,
                                            image_normalization=image_normalization,fix_imbalance=fix_imbalance,
                                           data_augmentation_rate=data_augmentation_rate)
    elif dataset=='hela10':
        return iterate_minibatches_hela10(inputs=inputs, targets=targets, batchsize=batchsize,
                                            shuffle=shuffle, augment=augment,yield_remaining=yield_remaining,
                                            image_normalization=image_normalization,fix_imbalance=fix_imbalance,
                                         data_augmentation_rate=data_augmentation_rate)
    elif dataset=='mnist':
        return iterate_minibatches_mnist(inputs=inputs, targets=targets, batchsize=batchsize,
                                            shuffle=shuffle, augment=augment,yield_remaining=yield_remaining)
    else:
        assert False
def iterate_minibatches_mnist(inputs, targets, batchsize,shuffle=False, augment=False,yield_remaining=False):
    assert len(inputs)== len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx=0

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        inp_exc = inputs[excerpt]
        yield inp_exc, targets[excerpt]


    if start_idx+batchsize<len(inputs) and yield_remaining:
        if shuffle:
            excerpt=indices[start_idx+batchsize:len(inputs)]
        else:
            excerpt=slice(start_idx+batchsize,len(inputs))
        inp_exc = inputs[excerpt]
        yield inp_exc,targets[excerpt]

def iterate_minibatches_cifar100(inputs, targets, batchsize, shuffle=False, augment=False,yield_remaining=False):
    assert len(inputs) == len(targets)
    batchsize=min(batchsize,len(inputs))
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(4,4),(4,4),(0,0)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                # Cropping and possible flipping
                if (np.random.randint(2) > 0):
                    random_cropped[r,:,:,:] = padded[r,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32),:]
                else:
                    random_cropped[r,:,:,:] = padded[r,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32),:][:,::-1,:]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]
        yield inp_exc, targets[excerpt]

    if start_idx+batchsize<len(inputs) and yield_remaining:
        if shuffle:
            excerpt=indices[start_idx+batchsize:len(inputs)]
        else:
            excerpt=slice(start_idx+batchsize,len(inputs))
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(4,4),(4,4),(0,0)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                # Cropping and possible flipping
                if (np.random.randint(2) > 0):
                    random_cropped[r,:,:,:] = padded[r,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32),:]
                else:
                    random_cropped[r,:,:,:] = padded[r,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32),:][:,::-1,:]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc,targets[excerpt]
def iterate_minibatches_breakhis(inputs, targets, batchsize, data_augmentation_rate,shuffle=False, augment=False,
                        yield_remaining=False,image_normalization=True,fix_imbalance=False):
    assert len(inputs) == len(targets)
    assert inputs.dtype==np.uint8

    if fix_imbalance:
        indices={}
        c=Counter(targets)
        for i in c.keys():
            indices[i]=np.where(targets==i)[0]
        max_class=max(c.values())
        for i in indices:
            if len(indices[i])<max_class:
                rep_num=int(max_class/len(indices[i]))
                remainder=max_class%len(indices[i])
                indices[i]=np.tile(indices[i],rep_num)
                if shuffle:
                    indices[i]=np.concatenate([indices[i],np.random.choice(indices[i],remainder)],axis=0)
                else:
                    indices[i]=np.concatenate([indices[i],indices[i][:remainder]],axis=0)
        indices=np.concatenate(list(indices.values()))
        cc=Counter(targets[indices])
        lens=list(cc.values())
        assert np.all(np.array(lens)==lens[0])
    else:
        indices=np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    batchsize=min(batchsize,len(indices))
    for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        if augment:
            random_cropped = np.zeros([len(excerpt),224,224,3], dtype=np.uint8)
            crops = np.random.random_integers(0,high=data_augmentation_rate,size=(batchsize,2))
            for r in range(batchsize):
                # Cropping and possible flipping
                resized=imresize(inputs[excerpt[r]],[224+data_augmentation_rate,224+data_augmentation_rate])
                if (np.random.randint(2) > 0):
                    random_cropped[r,:,:,:] = resized[crops[r,0]:(crops[r,0]+224),crops[r,1]:(crops[r,1]+224),:]
                else:
                    random_cropped[r,:,:,:] = resized[crops[r,0]:(crops[r,0]+224),crops[r,1]:(crops[r,1]+224),:][:,::-1,:]
            inp_exc = random_cropped
        else:
            resized=np.zeros([len(excerpt),224,224,3],dtype=np.uint8)
            for r in range(batchsize):
                resized[r,...]=imresize(inputs[excerpt[r]],[224,224,3])
            inp_exc = resized
        if image_normalization:
            inp_exc=normalize_image(inp_exc)
        yield inp_exc, targets[excerpt]
#         yield inp_exc, targets[excerpt],excerpt,indices

    if start_idx+batchsize<len(indices) and yield_remaining:
        num_remaining=len(indices)-(start_idx+batchsize)
        excerpt=indices[start_idx+batchsize:len(indices)]
        if augment:
            random_cropped = np.zeros([len(excerpt),224,224,3], dtype=np.uint8)
            crops = np.random.random_integers(0,high=data_augmentation_rate,size=(batchsize,2))
            for r in range(num_remaining):
                # Cropping and possible flipping
                resized=imresize(inputs[excerpt[r]],[224+data_augmentation_rate,224+data_augmentation_rate])
                if (np.random.randint(2) > 0):
                    random_cropped[r,:,:,:] = resized[crops[r,0]:(crops[r,0]+224),crops[r,1]:(crops[r,1]+224),:]
                else:
                    random_cropped[r,:,:,:] = resized[crops[r,0]:(crops[r,0]+224),crops[r,1]:(crops[r,1]+224),:][:,::-1,:]
            inp_exc = random_cropped
        else:
            resized=np.zeros([len(excerpt),224,224,3],dtype=np.uint8)
            for r in range(num_remaining):
                resized[r,...]=imresize(inputs[excerpt[r]],[224,224])
            inp_exc = resized
        if image_normalization:
            inp_exc=normalize_image(inp_exc)
        yield inp_exc, targets[excerpt]
def iterate_minibatches_hela10(inputs, targets, batchsize, data_augmentation_rate,shuffle=False, augment=False,
                        yield_remaining=False,image_normalization=True,fix_imbalance=False):
    assert len(inputs) == len(targets)
    assert inputs.dtype==np.int32

    if fix_imbalance:
        indices={}
        c=Counter(targets)
        for i in c.keys():
            indices[i]=np.where(targets==i)[0]
        max_class=max(c.values())
        for i in indices:
            if len(indices[i])<max_class:
                rep_num=int(max_class/len(indices[i]))
                remainder=max_class%len(indices[i])
                indices[i]=np.tile(indices[i],rep_num)
                if shuffle:
                    indices[i]=np.concatenate([indices[i],np.random.choice(indices[i],remainder)],axis=0)
                else:
                    indices[i]=np.concatenate([indices[i],indices[i][:remainder]],axis=0)
        indices=np.concatenate(list(indices.values()))
        cc=Counter(targets[indices])
        lens=list(cc.values())
        assert np.all(np.array(lens)==lens[0])
    else:
        indices=np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    batchsize=min(batchsize,len(indices))
    for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        if augment:
            random_cropped = np.zeros([len(excerpt),224,224,1], dtype=np.int32)
            crops = np.random.random_integers(0,high=data_augmentation_rate,size=(batchsize,2))
            for r in range(batchsize):
                # Cropping and possible flipping
                resized=imresize(inputs[excerpt[r]].squeeze(),[224+data_augmentation_rate,224+data_augmentation_rate])
                if (np.random.randint(2) > 0):
                    random_cropped[r,:,:,:] = resized[crops[r,0]:(crops[r,0]+224),crops[r,1]:(crops[r,1]+224)][:,:,np.newaxis]
                else:
                    random_cropped[r,:,:,:] = resized[crops[r,0]:(crops[r,0]+224),crops[r,1]:(crops[r,1]+224)][:,::-1,np.newaxis]
            inp_exc = random_cropped
        else:
            resized=np.zeros([len(excerpt),224,224,1],dtype=np.int32)
            for r in range(batchsize):
                resized[r,...]=imresize(inputs[excerpt[r]].squeeze(),[224,224])[:,:,np.newaxis]
            inp_exc = resized
        if image_normalization:
            inp_exc=normalize_grayscale_image(inp_exc)
        yield inp_exc, targets[excerpt]
#         yield inp_exc, targets[excerpt],excerpt,indices

    if start_idx+batchsize<len(indices) and yield_remaining:
        num_remaining=len(indices)-(start_idx+batchsize)
        excerpt=indices[start_idx+batchsize:len(indices)]
        if augment:
            random_cropped = np.zeros([len(excerpt),224,224,1], dtype=np.int32)
            crops = np.random.random_integers(0,high=data_augmentation_rate,size=(batchsize,2))
            for r in range(num_remaining):
                # Cropping and possible flipping
                resized=imresize(inputs[excerpt[r]].squeeze(),[224+data_augmentation_rate,224+data_augmentation_rate])
                if (np.random.randint(2) > 0):
                    random_cropped[r,:,:,:] = resized[crops[r,0]:(crops[r,0]+224),crops[r,1]:(crops[r,1]+224)][:,:,np.newaxis]
                else:
                    random_cropped[r,:,:,:] = resized[crops[r,0]:(crops[r,0]+224),crops[r,1]:(crops[r,1]+224)][:,::-1,np.newaxis]
            inp_exc = random_cropped
        else:
            resized=np.zeros([len(excerpt),224,224,1],dtype=np.int32)
            for r in range(num_remaining):
                resized[r,...]=imresize(inputs[excerpt[r]].squeeze(),[224,224])[:,:,np.newaxis]
            inp_exc = resized
        if image_normalization:
            inp_exc=normalize_grayscale_image(inp_exc)
        yield inp_exc, targets[excerpt]
def normalize_image(X):
    X=X/np.float32(255)
#     X_mean=np.mean(X,axis=0)
#     X-=X_mean
    return X
def normalize_grayscale_image(X):
    X=X/np.float32(X.max())

    return X.astype('float32')
def train_plain(X_train,Y_train,train_batch_size,train_tensors,session=None,dataset='cifar100'):
    if session is None:
        session=tf.get_default_session()
    assert session is not None
    evaluation_vals={}
    evaluation_vals['local_step']=0
    evaluation_vals['loss']=0
    evaluation_vals['class_loss']=0
    evaluation_vals['regularization_loss']=0
    evaluation_vals['top1_accuracy']=0
    evaluation_vals['top5_accuracy']=0
    num_batches_train=int((len(X_train)/train_batch_size))

    for X_minibatch,Y_minibatch in iterate_minibatches(X_train,Y_train,min(train_batch_size,len(X_train)),shuffle=True,augment=True,dataset=dataset):
        _,local_loss,local_class_loss,local_regularization_loss,local_top1_accuracy,local_top5_accuracy=\
        session.run([train_tensors['optimizer_train'],
                     train_tensors['loss_train'],
                     train_tensors['class_loss_train'],
                     train_tensors['regularization_loss_train'],
                     train_tensors['top1_accuracy_train'],
                     train_tensors['top5_accuracy_train']],
                  feed_dict={train_tensors['X_train']:X_minibatch,
                             train_tensors['Y_train']:Y_minibatch})
        weight=len(X_minibatch)/len(X_train)
        evaluation_vals['local_step']+=1
        evaluation_vals['loss']+=local_loss*weight
        evaluation_vals['class_loss']+=local_class_loss*weight
        evaluation_vals['regularization_loss']+=local_regularization_loss*weight
        evaluation_vals['top1_accuracy']+=local_top1_accuracy*weight
        evaluation_vals['top5_accuracy']+=local_top5_accuracy*weight
    return evaluation_vals
def train_with_sample_weight(X_train,Y_train,sample_weight_train,train_batch_size,train_tensors,session=None,dataset='cifar100'):
    if session is None:
        session=tf.get_default_session()
    assert session is not None
    assert len(X_train)==len(Y_train)
    assert len(X_train)==len(sample_weight_train)
    evaluation_vals={}
    evaluation_vals['local_step']=0
    evaluation_vals['loss']=0
    evaluation_vals['class_loss']=0
    evaluation_vals['regularization_loss']=0
    evaluation_vals['top1_accuracy']=0
    evaluation_vals['top5_accuracy']=0
    num_batches_train=int((len(X_train)/train_batch_size))
    indices_train=np.arange(len(Y_train))
    for X_minibatch,indices_minibatch in iterate_minibatches(X_train,indices_train,min(train_batch_size,len(X_train)),shuffle=True,augment=True,dataset=dataset):
        _,local_loss,local_class_loss,local_regularization_loss,local_top1_accuracy,local_top5_accuracy=\
        session.run([train_tensors['optimizer_train'],
                     train_tensors['loss_train'],
                     train_tensors['class_loss_train'],
                     train_tensors['regularization_loss_train'],
                     train_tensors['top1_accuracy_train'],
                     train_tensors['top5_accuracy_train']],
                  feed_dict={train_tensors['X_train']:X_minibatch,
                             train_tensors['Y_train']:Y_train[indices_minibatch],
                             train_tensors['sample_weight_train']:sample_weight_train[indices_minibatch]})
        if local_loss>100 or np.isnan(local_loss):
            print(local_loss)
        weight=len(X_minibatch)/len(X_train)
        evaluation_vals['local_step']+=1
        evaluation_vals['loss']+=local_loss*weight
        evaluation_vals['class_loss']+=local_class_loss*weight
        evaluation_vals['regularization_loss']+=local_regularization_loss*weight
        evaluation_vals['top1_accuracy']+=local_top1_accuracy*weight
        evaluation_vals['top5_accuracy']+=local_top5_accuracy*weight
    return evaluation_vals
def get_loss_value(X_test,Y_test,test_batch_size,evaluation_tensors,session=None,dataset='cifar100'):
    if session is None:
        session=tf.get_default_session()
    assert session is not None
    loss_values=[]
    for X_minibatch,Y_minibatch in iterate_minibatches(X_test,Y_test,test_batch_size,shuffle=False,
                                                       augment=False,yield_remaining=True,fix_imbalance=False,dataset=dataset):
        loss_value,=\
        session.run([evaluation_tensors['batch_loss']],feed_dict={evaluation_tensors['X']:X_minibatch,
                                                                  evaluation_tensors['Y']:Y_minibatch})
        loss_values.append(loss_value)
    return np.concatenate(loss_values,axis=0)

def train_distillation(X_train,Y_train,iteration,hyper_params,fixed_params,train_tensors,session=None,
                       previous_network=None,class_ord=None,dataset='cifar100'):
    eval_vals={}
    eval_vals['local_step']=0
    eval_vals['loss']=0
    eval_vals['class_loss']=0
    eval_vals['regularization_loss']=0
    eval_vals['top1_accuracy']=0
    eval_vals['top5_accuracy']=0
    if session is None:
        session=tf.get_default_session()
    assert session is not None
    num_batches_train=int((len(X_train)/hyper_params['train_batch_size']))
    if iteration>=1:
        assert previous_network is not None and class_ord is not None
    for X_minibatch,Y_minibatch in iterate_minibatches(X_train,Y_train,hyper_params['train_batch_size'],
                                                       shuffle=True,augment=True,dataset=dataset):
        Y_one_hot_minibatch=np.zeros([len(Y_minibatch),fixed_params['total_num_classes']],dtype=np.float32)
        Y_one_hot_minibatch[range(len(Y_minibatch)),Y_minibatch]=1.
        #distillation
        if iteration>=1:
            prediction_old=session.run([train_tensors['network_prev_sigmoid']], #!!!do not create tensor here!!!
                                        feed_dict={previous_network.tf_tensors['input']:X_minibatch})[0]
            Y_one_hot_minibatch[:,class_ord[:iteration*fixed_params['class_batch_size']]]=\
            prediction_old[:,class_ord[:iteration*fixed_params['class_batch_size']]]
        _,local_loss,local_class_loss,local_regularization_loss,local_top1_accuracy,local_top5_accuracy=\
        session.run([train_tensors['optimizer_train'],
                     train_tensors['loss_train'],
                     train_tensors['class_loss_train'],
                     train_tensors['regularization_loss_train'],
                     train_tensors['top1_accuracy_train'],
                     train_tensors['top5_accuracy_train']],
                  feed_dict={train_tensors['X_train']:X_minibatch,
                             train_tensors['Y_one_hot_train']:Y_one_hot_minibatch,
                             train_tensors['Y_train']:Y_minibatch})
        weight=len(X_minibatch)/len(X_train)
        eval_vals['local_step']+=1
        eval_vals['loss']+=local_loss*weight
        eval_vals['class_loss']+=local_class_loss*weight
        eval_vals['regularization_loss']+=local_regularization_loss*weight
        eval_vals['top1_accuracy']+=local_top1_accuracy*weight
        eval_vals['top5_accuracy']+=local_top5_accuracy*weight
    return eval_vals
def train_distillation_and_ground_truth(X_train,Y_train,iteration,hyper_params,
                                        fixed_params,train_tensors,session=None,previous_network=None,
                                        class_ord=None,dataset='cifar100'):
    if session is None:
        session=tf.get_default_session()
    assert session is not None
    eval_vals={}
    eval_vals['local_step']=0
    eval_vals['loss']=0
    eval_vals['class_loss']=0
    eval_vals['regularization_loss']=0
    eval_vals['top1_accuracy']=0
    eval_vals['top5_accuracy']=0
    num_batches_train=int((len(X_train)/hyper_params['train_batch_size']))
    if iteration>=1:
        assert previous_network is not None and class_ord is not None
        boundary=iteration*fixed_params['class_batch_size']
    for X_minibatch,Y_minibatch in iterate_minibatches(X_train,Y_train,hyper_params['train_batch_size'],
                                                       shuffle=True,augment=True,dataset=dataset):
        Y_one_hot_minibatch=np.zeros([len(Y_minibatch),fixed_params['total_num_classes']],dtype=np.float32)
        Y_one_hot_minibatch[range(len(Y_minibatch)),Y_minibatch]=1.
        #distillation
        if iteration>=1:
            prediction_old=session.run([train_tensors['network_prev_sigmoid']], #!!!do not create tensor here!!!
                                        feed_dict={previous_network.tf_tensors['input']:X_minibatch})[0]
            #use numpy `gather_nd`
            ind=np.where(Y_minibatch>=boundary)[0][:,np.newaxis]
            Y_one_hot_minibatch[ind,class_ord[:boundary]]=\
            prediction_old[ind,class_ord[:boundary]]
        _,local_loss,local_class_loss,local_regularization_loss,local_top1_accuracy,local_top5_accuracy=\
        session.run([train_tensors['optimizer_train'],
                     train_tensors['loss_train'],
                     train_tensors['class_loss_train'],
                     train_tensors['regularization_loss_train'],
                     train_tensors['top1_accuracy_train'],
                     train_tensors['top5_accuracy_train']],
                  feed_dict={train_tensors['X_train']:X_minibatch,
                             train_tensors['Y_one_hot_train']:Y_one_hot_minibatch,
                             train_tensors['Y_train']:Y_minibatch})

        eval_vals['local_step']+=1
        eval_vals['loss']+=local_loss/num_batches_train
        eval_vals['class_loss']+=local_class_loss/num_batches_train
        eval_vals['regularization_loss']+=local_regularization_loss/num_batches_train
        eval_vals['top1_accuracy']+=local_top1_accuracy/num_batches_train
        eval_vals['top5_accuracy']+=local_top5_accuracy/num_batches_train
    return eval_vals
def initialize_exemplars_and_exemplars_mean(W,H,C,total_num_classes,exemplars_set_size,feat_size,dataset,use_theoretical_mean):
    icarl_exemplars,svm_exemplars=initialize_exemplars(W,H,C,total_num_classes,exemplars_set_size,dataset)
    icarl_exemplars_mean=initialize_icarl_exemplars_mean(total_num_classes,feat_size)
    if use_theoretical_mean:
        theoretical_mean=initialize_icarl_theoretical_mean(total_num_classes,feat_size)
        return icarl_exemplars,svm_exemplars,icarl_exemplars_mean,theoretical_mean
    else:
        return icarl_exemplars,svm_exemplars,icarl_exemplars_mean
def initialize_exemplars(W,H,C,total_num_classes,exemplars_set_size,dataset):
    if dataset in ['cifar100','cifar10','mnist']:
        icarl_exemplars=np.ones([total_num_classes,exemplars_set_size,W,H,C])*np.nan
        svm_exemplars=np.ones([total_num_classes,exemplars_set_size,W,H,C])*np.nan
    elif dataset=='breakhis':
        icarl_exemplars=np.ones([total_num_classes,exemplars_set_size,W,H,C]).astype('uint8')
        svm_exemplars=np.ones([total_num_classes,exemplars_set_size,W,H,C]).astype('uint8')
    elif dataset=='hela10':
        icarl_exemplars=np.ones([total_num_classes,exemplars_set_size,W,H,C]).astype('int32')
        svm_exemplars=np.ones([total_num_classes,exemplars_set_size,W,H,C]).astype('int32')
    return icarl_exemplars,svm_exemplars
def initialize_icarl_exemplars_mean(total_num_classes,feat_size):
    return np.ones([total_num_classes,feat_size])*np.nan
def initialize_icarl_theoretical_mean(total_num_classes,feat_size):
    return np.ones([total_num_classes,feat_size])*np.nan
def update_exemplars_set_icarl(exemplars_set,data_dict,class_ord,classes_ind_to_update,batch_size,exemplars_per_class,
                           feature_map_tensors,session,dataset='cifar100'):
    assert len(data_dict['X_train'])==len(data_dict['Y_train'])
    for i in classes_ind_to_update:
        class_ind=np.where(data_dict['Y_train']==class_ord[i])[0]
        X_train_class=data_dict['X_train'][class_ind]
        Y_train_class=data_dict['Y_train'][class_ind]
        if len(X_train_class)<exemplars_per_class:
            print('Requiring %d exemplars, but only %d examples of class %d in training set'%(exemplars_per_class,len(X_train_class),class_ord[i]))
            class_ind_chosen=np.random.choice(np.arange(len(X_train_class)),exemplars_per_class,replace=True)
            exemplars_set[i,...]=X_train_class[class_ind_chosen,...]
        else:
            feature_maps=test_get_feature_maps(X_train_class,batch_size,feature_map_tensors,session,dataset=dataset) #batch_size*64
            D=feature_maps.T#64*batch_size
            D/=np.linalg.norm(D,axis=0)#64*batch_size
            mu=np.mean(D,axis=1)#64
            w_t=mu.copy() # must get a copy of mu array

            for e in range(exemplars_per_class):
                tmp_t=np.dot(w_t,D)
                ind_max=np.nanargmax(tmp_t)
                exemplars_set[i,e,...]=X_train_class[ind_max,...]
                w_t+=mu-D[:,ind_max]
                D[:,ind_max]=np.nan
            assert np.sum(np.all(np.isnan(D),axis=0))==exemplars_per_class
# to be updated
def update_exemplars_set_by_loss(exemplars_set,data_dict,class_list,batch_size,exemplars_per_class,
                                evaluation_tensors,session,base_dir=None,dataset='cifar100'):
    assert len(data_dict['X_train'])==len(data_dict['Y_train'])
    for i,cl in enumerate(class_list):
        class_ind=np.where(data_dict['Y_train']==cl)[0]
        X_train_class=data_dict['X_train'][class_ind]
        Y_train_class=data_dict['Y_train'][class_ind]
        if len(X_train_class)<exemplars_per_class:
            print('Requiring %d exemplars, but only %d examples of class %d in ing set'%(exemplars_per_class,len(X_train_class),cl))
            class_ind_chosen=np.random.choice(np.arange(len(X_train_class)),exemplars_per_class,replace=True)
            exemplars_set[i,...]=X_train_class[class_ind_chosen,...]
        else:
            loss_values=get_loss_value(X_train_class,np.ones(len(X_train_class))*cl,batch_size,evaluation_tensors,
                                       session,dataset=dataset) #batch_size*64
            #dump loss_value for analysis
            if base_dir is not None:
                with open(os.path.join(base_dir,'loss_value_class_%d.pkl'%cl),'wb') as f:
                    pickle.dump(loss_values,f)
            argsort_res=loss_values.argsort()
            start_idx=int((len(argsort_res)-exemplars_per_class)/2)
            exemplars_set[cl,...]=X_train_class[argsort_res[start_idx:start_idx+exemplars_per_class]]
            assert np.all(~np.isnan(exemplars_set[cl,...]))
    return exemplars_set
# to be updated
def update_exemplars_set_by_svm(exemplars_set,data_dict,class_list,batch_size,exemplars_per_class,
                           feature_map_tensors,session,dataset='cifar100'):
    feature_maps_train=test_get_feature_maps(data_dict['X_train'],batch_size,feature_map_tensors,session,dataset=dataset)
    svm=SVC()
    svm.fit(feature_maps_train,data_dict['Y_train'])
    svm_support_cumsum=np.hstack([0,svm.n_support_]).cumsum()
    for i,cl in enumerate(class_list):
        svm_class_ind=list(svm.classes_).index(cl)
        support_vector_indices=svm.support_[svm_support_cumsum[svm_class_ind]:svm_support_cumsum[svm_class_ind+1]]
        assert len(support_vector_indices)==svm.n_support_[svm_class_ind]
        repeat=int(exemplars_per_class/len(support_vector_indices))
        remainder=exemplars_per_class%len(support_vector_indices)
        repeat_indices=np.tile(support_vector_indices,[repeat])
        remainder_indices=np.random.choice(support_vector_indices,remainder,replace=False)
        support_vector_indices_selected=np.concatenate([repeat_indices,remainder_indices])
        exemplars_set[cl,...]=data_dict['X_train'][support_vector_indices_selected]
    return exemplars_set,svm
def get_fixedsize_exemplars_set_icarl(data_dict_cumul,exemplars_set_size,
                               test_batch_size,feature_map_tensors,class_ord,classes_ind_up_to_now,session,dataset='cifar100'):
    classes_up_to_now=[class_ord[i] for i in classes_ind_up_to_now]
    assert len(data_dict_cumul['X_train'])==len(data_dict_cumul['Y_train'])
    assert exemplars_set_size>=len(classes_up_to_now)
    exemplars_per_class=int(exemplars_set_size/len(classes_up_to_now))
    _,W,H,C=data_dict_cumul['X_train'].shape
    feat_size=feature_map_tensors['feature_map'].shape.as_list()[1]
    exemplars_set,_,_=initialize_exemplars_and_exemplars_mean(W,H,C,
                                          len(classes_up_to_now),exemplars_per_class,feat_size,dataset=dataset,use_theoretical_mean=False)
    for i,cl in enumerate(classes_up_to_now):
        class_ind=np.where(data_dict_cumul['Y_train']==cl)[0]
        X_train_class=data_dict_cumul['X_train'][class_ind]
        Y_train_class=data_dict_cumul['Y_train'][class_ind]
        if len(X_train_class)<exemplars_per_class:
            print('Requiring %d exemplars, but only %d examples of class %d in training set'%(exemplars_per_class,len(X_train_class),cl))
            class_ind_chosen=np.random.choice(np.arange(len(X_train_class)),exemplars_per_class,replace=True)
            exemplars_set[i,...]=X_train_class[class_ind_chosen,...]
        else:
            feature_maps=test_get_feature_maps(X_train_class,test_batch_size,feature_map_tensors,session,dataset=dataset) #batch_size*64
            D=feature_maps.T#64*batch_size
            D/=np.linalg.norm(D,axis=0)#64*batch_size
            mu=np.mean(D,axis=1)#64
            w_t=mu.copy() # must get a copy of mu array

            for e in range(exemplars_per_class):
                tmp_t=np.dot(w_t,D)
                ind_max=np.nanargmax(tmp_t)
                exemplars_set[i,e,...]=X_train_class[ind_max,...]
                w_t+=mu-D[:,ind_max]
                D[:,ind_max]=np.nan
            assert np.sum(np.all(np.isnan(D),axis=0))==exemplars_per_class
    return exemplars_set
def get_fixed_size_exemplars_set_by_svm(data_dict_cumul,exemplars_set_size,test_batch_size,feature_map_tensors,classes_up_to_now,
                                       session,dataset='cifar100'):
    assert len(data_dict_cumul['X_train'])==len(data_dict_cumul['Y_train'])
    assert exemplars_set_size>=len(classes_up_to_now)
    exemplars_per_class=int(exemplars_set_size/len(classes_up_to_now))
    _,W,H,C=data_dict_cumul['X_train'].shape
    feat_size=feature_map_tensors['feature_map'].shape.as_list()[1]
    exemplars_set,_,_=initialize_exemplars_and_exemplars_mean(W,H,C,
                                          len(classes_up_to_now),exemplars_per_class,feat_size,dataset=dataset,use_theoretical_mean=False)
    feature_maps_train=test_get_feature_maps(data_dict_cumul['X_train'],test_batch_size,feature_map_tensors,session,dataset=dataset)
    svm=SVC()
    svm.fit(feature_maps_train,data_dict_cumul['Y_train'])
    svm_support_cumsum=np.hstack([0,svm.n_support_]).cumsum()
    for i,cl in enumerate(classes_up_to_now):
        svm_class_ind=list(svm.classes_).index(cl)
        support_vector_indices=svm.support_[svm_support_cumsum[svm_class_ind]:svm_support_cumsum[svm_class_ind+1]]
        assert len(support_vector_indices)==svm.n_support_[svm_class_ind]
        repeat=int(exemplars_per_class/len(support_vector_indices))
        remainder=exemplars_per_class%len(support_vector_indices)
        repeat_indices=np.tile(support_vector_indices,[repeat])
        remainder_indices=np.random.choice(support_vector_indices,remainder,replace=False)
        support_vector_indices_selected=np.concatenate([repeat_indices,remainder_indices])
        exemplars_set[i,...]=data_dict_cumul['X_train'][support_vector_indices_selected]
    return exemplars_set,svm
def get_fixed_size_exemplars_set_by_svm_without_repeat(data_dict_cumul,
                                                      exemplars_set_size,test_batch_size,feature_map_tensors,class_ord,classes_ind_to_update,
                                                       session,dataset='cifar100'):
    classes_up_to_now=[class_ord[i] for i in classes_ind_to_update]
    assert len(data_dict_cumul['X_train'])==len(data_dict_cumul['Y_train'])
    assert exemplars_set_size>=len(classes_up_to_now)
    exemplars_per_class=int(exemplars_set_size/len(classes_up_to_now))
    _,W,H,C=data_dict_cumul['X_train'].shape
    feat_size=feature_map_tensors['feature_map'].shape.as_list()[1]
    exemplars_set,_,_=initialize_exemplars_and_exemplars_mean(W,H,C,
                                          len(classes_up_to_now),exemplars_per_class,feat_size,dataset=dataset,use_theoretical_mean=False)
    feature_maps_train=test_get_feature_maps(data_dict_cumul['X_train'],test_batch_size,feature_map_tensors,session,dataset=dataset)
    svm=SVC()
    svm.fit(feature_maps_train,data_dict_cumul['Y_train'])
    svm_support_cumsum=np.hstack([0,svm.n_support_]).cumsum()
    for i,cl in enumerate(classes_up_to_now):
        class_ind=np.where(data_dict_cumul['Y_train']==cl)[0]
        svm_class_ind=list(svm.classes_).index(cl)
        support_vector_indices=svm.support_[svm_support_cumsum[svm_class_ind]:svm_support_cumsum[svm_class_ind+1]]
        print('%d support_vectors for class %d'%(len(support_vector_indices),cl))
        assert len(support_vector_indices)==svm.n_support_[svm_class_ind]
        if exemplars_per_class<=len(support_vector_indices):
            indices_selected=np.random.choice(support_vector_indices,exemplars_per_class,replace=False)
        else:
            print('support vector not enough for class %d, sampling from previous training set'%(cl))
            others=np.random.choice(class_ind,exemplars_per_class-len(support_vector_indices),replace=True)
            indices_selected=np.concatenate([support_vector_indices,others])
        exemplars_set[i,...]=data_dict_cumul['X_train'][indices_selected]
    return exemplars_set,svm


def update_exemplars_mean_icarl(icarl_exemplars_mean,icarl_exemplars,test_batch_size,feature_map_tensors,use_fixedsize_exemplars,class_ord=None,classes_ind_to_update=None,session=None,dataset='cifar100'):
    session=assert_session(session)
    if use_fixedsize_exemplars:
        current_exemplars=icarl_exemplars
    else:
        assert classes_ind_to_update is not None
        current_exemplars=icarl_exemplars[classes_ind_to_update,...]
    CL,PR,W,H,C=current_exemplars.shape
    assert(np.all(~np.isnan(current_exemplars)))
    feature_maps=test_get_feature_maps(current_exemplars.reshape([-1,W,H,C]),
                                                test_batch_size,
                                       feature_map_tensors,
                                       session,shuffle=False,yield_remaining=True,dataset=dataset)\
    .reshape([CL,PR,-1])
    unnormalized_exemplars_mean=feature_maps.mean(axis=1)
    icarl_exemplars_mean[classes_ind_to_update,...]=\
    unnormalized_exemplars_mean/np.linalg.norm(unnormalized_exemplars_mean,axis=1)[...,np.newaxis]
    return icarl_exemplars_mean
def update_theoretical_mean_icarl(theoretical_mean,data_dict_total,test_batch_size,feature_map_tensors,class_ord,classes_ind_to_update,session=None,dataset='cifar100'):
    session=assert_session(session)
    classes_to_update=[class_ord[j] for j in classes_ind_to_update]
    prev_ind=[i in classes_to_update for i in data_dict_total['Y_train']]
    X_prev=data_dict_total['X_train'][prev_ind]
    Y_prev=data_dict_total['Y_train'][prev_ind]
    feature_maps=test_get_feature_maps(X_prev,test_batch_size,
                                       feature_map_tensors,session,shuffle=False,yield_remaining=True,dataset=dataset)

    for i in classes_ind_to_update:
        unnormalized_theoretical_mean=feature_maps[Y_prev==class_ord[i]].mean(axis=0)
        theoretical_mean[i,:]=unnormalized_theoretical_mean/np.linalg.norm(unnormalized_theoretical_mean)
    return theoretical_mean

def exemplars_as_training_set(exemplars,use_fixedsize_exemplars,class_ord,classes_ind_up_to_now):
    n_CL,n_E,W,H,C=exemplars.shape
    if use_fixedsize_exemplars:
        exemplars_up_to_now=exemplars.reshape((-1,W,H,C))
    else:
        exemplars_up_to_now=exemplars[classes_ind_up_to_now,...].reshape((-1,W,H,C))
    assert exemplars_up_to_now.shape[0]==len(classes_ind_up_to_now)*n_E
    assert np.all(~np.isnan(exemplars_up_to_now))
    classes_up_to_now=[class_ord[i] for i in classes_ind_up_to_now]
    exemplars_up_to_now_label=np.tile(np.array(classes_up_to_now).reshape(len(classes_up_to_now),1),[1,n_E]).reshape(-1,1).squeeze()
    return exemplars_up_to_now,exemplars_up_to_now_label


def test_get_feature_maps(X,batch_size,feature_map_tensors,session=None,shuffle=False,yield_remaining=True,dataset='cifar100'):
    if session is None:
        session=tf.get_default_session()
    assert session is not None
    feature_maps=[]
    Y=np.ones(len(X))*np.nan #dummy Y
    for X_minibatch,Y_minibatch in iterate_minibatches(X,Y,batch_size,shuffle=shuffle,augment=False,
                                                       yield_remaining=yield_remaining,
                                                       fix_imbalance=False,dataset=dataset):
        feature_map=session.run([feature_map_tensors['feature_map']],
                                 feed_dict={feature_map_tensors['X']:X_minibatch})
        feature_maps.append(feature_map[0])
    feature_maps=np.concatenate(feature_maps,axis=0)
    return feature_maps
def test_ncm_dist(X_test,class_means,test_batch_size,evaluation_tensors,session,dataset='cifar100'):
    rank_of_classes=np.zeros([len(X_test),class_means.shape[1]])
    feature_map_tensors=dict(X=evaluation_tensors['X'],
                            feature_map=evaluation_tensors['feature_map'])
    feature_maps=test_get_feature_maps(X_test,test_batch_size,feature_map_tensors,session,dataset=dataset)
    pred_inter=(feature_maps.T/np.linalg.norm(feature_maps.T,axis=0)).T
    pred_inter=pred_inter/np.linalg.norm(pred_inter,axis=1)[...,np.newaxis]
    sqd=cdist(class_means,pred_inter,'sqeuclidean').T
    return sqd
def test_accuracy_evaluation_ncm(X_test,Y_test,class_means,test_batch_size,evaluation_tensors,session,dataset='cifar100'):
    assert(len(X_test)==len(Y_test))
    sqd=test_ncm_dist(X_test,class_means,test_batch_size,evaluation_tensors,session,dataset=dataset)
    rank_of_classes=sqd.argsort(axis=1).argsort(axis=1)
    rank_of_ground_truth_class=rank_of_classes[range(len(Y_test)),Y_test]
    evaluation_vals=dict(top1_accuracy=np.mean(rank_of_ground_truth_class<1),
                         top5_accuracy=np.mean(rank_of_ground_truth_class<5),
                         rank_of_ground_truth_class=rank_of_ground_truth_class)

    Y_pred=np.nanargmin(sqd,axis=1)
    sklearn_eval_vals=sklearn_evaluation(Y_test,Y_pred)
    evaluation_vals.update(sklearn_eval_vals)

    return evaluation_vals
def test_prediction_ncm(X_test,class_means,test_batch_size,evaluation_tensors,session,dataset='cifar100'):
    sqd=test_ncm_dist(X_test,class_means,test_batch_size,evaluation_tensors,session,dataset=dataset)
    pred=sqd.argmin(axis=1).astype(np.int32)
    return pred
def update_exemplars_mean(exemplars_mean,exemplars,
                         class_batch_size,test_batch_size,iteration,class_ord,feature_map_tensors,session,dataset='cifar100'):
    current_exemplars=exemplars[class_ord[:((iteration+1)*class_batch_size)]]
    CL,PR,W,H,C=current_exemplars.shape
    assert(np.all(~np.isnan(current_exemplars)))
    feature_maps=test_get_feature_maps(current_exemplars.reshape([-1,W,H,C]),
                                                test_batch_size,
                                       feature_map_tensors,
                                       session,shuffle=False,yield_remaining=True,dataset=dataset)\
    .reshape([CL,PR,-1])
    unnormalized_exemplars_mean=feature_maps.mean(axis=1)
    exemplars_mean[class_ord[:(iteration+1)*class_batch_size]]=\
    unnormalized_exemplars_mean/np.linalg.norm(unnormalized_exemplars_mean,axis=1)[...,np.newaxis]
    return exemplars_mean
def update_theoretical_mean(theoretical_mean,X_train,Y_train,class_batch_size,test_batch_size,
                            iteration,class_ord,feature_map_tensors,session,dataset='cifar100'):
    prev_classes=class_ord[:((iteration+1)*class_batch_size)]
    prev_ind=[i in prev_classes for i in Y_train]
    X_prev=X_train[prev_ind]
    Y_prev=Y_train[prev_ind]
    feature_maps=test_get_feature_maps(X_prev,test_batch_size,
                                       feature_map_tensors,session,shuffle=False,yield_remaining=True,dataset=dataset)
    for c in prev_classes:
        unnormalized_theoretical_mean=feature_maps[Y_prev==c].mean(axis=0)
        theoretical_mean[c,:]=unnormalized_theoretical_mean/np.linalg.norm(unnormalized_theoretical_mean)
    return theoretical_mean
def last_layer_retrain(X_train,Y_train,train_batch_size,
                       train_tensors,train_network,best_model_file,num_epochs,train_method,session=None,dataset='cifar100'):
    if session==None:
        session=tf.get_default_session()
    assert session is not None
    train_network.load_model(best_model_file,session)
    train_tensors=copy(train_tensors)
    train_tensors['optimizer_train']=train_tensors['optimizer_fc_train']
    train_eval_vals_list=[]
    for epoch in range(num_epochs):
        print('Epoch %d'%(epoch+1))
        if train_method=='train_with_sample_weight':
            train_eval_vals=train_with_sample_weight(X_train,Y_train,
                                                     np.ones_like(Y_train),train_batch_size,
                                                     train_tensors,session,dataset=dataset)

        elif train_method in ['train_distillation_and_ground_truth','train_plain','train_distillation']:
            train_eval_vals=train_plain(X_train,
                                        Y_train,
                                        train_batch_size,
                                        train_tensors,session,dataset=dataset)
        train_eval_vals_list.append(train_eval_vals)
    return train_eval_vals_list
def last_layer_retrain_with_exemplars(exemplars,class_ord,classes_ind_up_to_now,train_batch_size,
                                      train_tensors,train_network,best_model_file,num_epochs,train_method,use_fixedsize_exemplars,session=None,dataset='cifar100'):
    exemplars_up_to_now,exemplars_up_to_now_label=exemplars_as_training_set(exemplars,use_fixedsize_exemplars,class_ord,classes_ind_up_to_now)
    train_eval_vals_list=last_layer_retrain(exemplars_up_to_now,exemplars_up_to_now_label,train_batch_size,
                       train_tensors,train_network,best_model_file,num_epochs,train_method,session=session,dataset=dataset)
    return train_eval_vals_list
def svm_retrain_with_exemplars(exemplars,class_ord,classes_ind_up_to_now,batch_size,feature_map_tensors,use_fixedsize_exemplars,session=None,dataset='cifar100'):
    if session==None:
        session=tf.get_default_session()
    assert session is not None
    exemplars_up_to_now,exemplars_up_to_now_label=exemplars_as_training_set(exemplars,use_fixedsize_exemplars,class_ord,classes_ind_up_to_now)
    feature_maps=test_get_feature_maps(exemplars_up_to_now,batch_size,feature_map_tensors,session=session,dataset=dataset)
    svm=SVC()
    svm.fit(feature_maps,exemplars_up_to_now_label)
    return svm
def test_prediction_svm(X_test,test_batch_size,feature_map_tensors,svm_model,session=None,dataset='cifar100'):
    if session==None:
        session=tf.get_default_session()
    assert session is not None
    feature_maps=test_get_feature_maps(X_test,test_batch_size,feature_map_tensors,session=session,dataset=dataset)
    pred_svm=svm_model.predict(feature_maps)
    return pred_svm
def test_evaluation_svm(X_test,Y_test,test_batch_size,feature_map_tensors,svm_model,session=None,dataset='cifar100'):
    if session==None:
        session=tf.get_default_session()
    assert session is not None
    eval_vals={}
    pred_svm=test_prediction_svm(X_test,test_batch_size,feature_map_tensors,svm_model,session=session,dataset=dataset)
    eval_vals=sklearn_evaluation(Y_test,pred_svm)
    return eval_vals
def sklearn_evaluation(label,predicted_label):
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    conf_matrix=confusion_matrix(label,predicted_label)
    report=classification_report(label,predicted_label)
    acc_score=accuracy_score(label,predicted_label)
    ck_score=cohen_kappa_score(label,predicted_label)
    precision_micro=metrics.precision_score(label, predicted_label, average='micro')
    precision_macro=metrics.precision_score(label,predicted_label,average='macro')
    recall_micro=metrics.recall_score(label,predicted_label,average='micro')
    recall_macro=metrics.recall_score(label,predicted_label,average='macro')
    f1_micro=metrics.f1_score(label,predicted_label,average='micro')
    f1_macro=metrics.f1_score(label,predicted_label,average='macro')
    return dict(confusion_matrix=conf_matrix,
                report_string=report,
                top1_accuracy=acc_score,
                ck_score=ck_score,
                precision_micro=precision_micro,
                precision_macro=precision_macro,
                recall_micro=recall_micro,
                recall_macro=recall_macro,
                f1_micro=f1_micro,
                f1_macro=f1_macro)

def data_dict_each_iteration(base_dir,num_iterations,class_batch_size,data_dict_total):
    data_dict_iterations=[]
    class_ord_file=os.path.join(base_dir,"class_ord.json")
    with open(class_ord_file,'r') as f:
        class_ord=json.load(f)
    for iteration in range(num_iterations):
        using_classes=class_ord[iteration*class_batch_size:(iteration+1)*class_batch_size]
        train_idx=np.array([i in using_classes for i in data_dict_total['Y_train']])
        test_idx=np.array([i in using_classes for i in data_dict_total['Y_test']])
        data_dict=dict(X_train=data_dict_total['X_train'][train_idx],
                       Y_train=data_dict_total['Y_train'][train_idx],
                       X_test=data_dict_total['X_test'][test_idx],
                       Y_test=data_dict_total['Y_test'][test_idx])
        data_dict_iterations.append(data_dict)
    return data_dict_iterations

def eval_vals_each_iteration(base_dir,num_iterations,class_batch_size,test_batch_size,tf_tensors,tf_networks,data_dict_iterations,dataset,session):
    plain_evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                            class_loss=tf_tensors['class_loss_test'],
                            regularization_loss=tf_tensors['regularization_loss_test'],
                            top1_accuracy=tf_tensors['top1_accuracy_test'],
                            top5_accuracy=tf_tensors['top5_accuracy_test'],
                            X=tf_tensors['X_test'],
                            Y=tf_tensors['Y_test'],
                            fc=tf_networks['network_test'].tf_tensors['fc'])
    eval_vals_lists=list()
    for iteration in range(num_iterations):
        eval_vals_list=list()
        final_best_model_params_file=os.path.join(base_dir,'best_model_params_%d.pkl'%(iteration))
        tf_networks['network_test'].load_model(final_best_model_params_file)    
        tf_networks['network_test'].reset_ewc_variables(session) # should not have ewc at final evaluation time  
        for data_dict_idx in range(num_iterations):
            data_dict=data_dict_iterations[data_dict_idx]
            eval_vals_test=test_accuracy_evaluation_plain(data_dict['X_test'],
                                                   data_dict['Y_test'],
                                                   test_batch_size,
                                                   plain_evaluation_tensors,session,dataset=dataset)
            eval_vals_list.append(eval_vals_test)
        eval_vals_lists.append(eval_vals_list)
    
    return eval_vals_lists
def table2(base_dir,first_iteration,last_iteration,class_batch_size,test_batch_size,tf_tensors,tf_networks,data_dict_full,data_dict_iterations,dataset,session):
    plain_evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                            class_loss=tf_tensors['class_loss_test'],
                            regularization_loss=tf_tensors['regularization_loss_test'],
                            top1_accuracy=tf_tensors['top1_accuracy_test'],
                            top5_accuracy=tf_tensors['top5_accuracy_test'],
                            X=tf_tensors['X_test'],
                            Y=tf_tensors['Y_test'],
                            fc=tf_networks['network_test'].tf_tensors['fc'])
    final_best_model_params_file=os.path.join(base_dir,'best_model_params_%d.pkl'%(last_iteration))
    tf_networks['network_test'].load_model(final_best_model_params_file)    
    tf_networks['network_test'].reset_ewc_variables(session) # should not have ewc at final evaluation time      
    eval_vals_real_data=test_accuracy_evaluation_plain(data_dict_iterations[last_iteration]['X_train'],
                                                        data_dict_iterations[last_iteration]['Y_train'],
                                                        test_batch_size,
                                                        plain_evaluation_tensors,session,dataset=dataset)
    eval_vals_all_data=test_accuracy_evaluation_plain(data_dict_full['X_train'],
                                                        data_dict_full['Y_train'],
                                                        test_batch_size,
                                                        plain_evaluation_tensors,session,dataset=dataset)
    eval_vals_test_data= test_accuracy_evaluation_plain(data_dict_iterations[last_iteration]['X_train'],
                                                        data_dict_iterations[last_iteration]['Y_train'],
                                                        test_batch_size,
                                                        plain_evaluation_tensors,session,dataset=dataset)       
    eval_vals=dict(real_data=eval_vals_real_data,all_data=eval_vals_all_data,test_data=eval_vals_test_data)                                                                
    return eval_vals

def feature_each_iteration(base_dir,num_iterations,test_batch_size,tf_tensors,tf_networks,X,dataset,session):
    feature_map_tensors=dict(X=tf_tensors['X_test'],feature_map=tf_tensors['feature_map_test'])
    feature_map_list=[]
    for iteration in range(num_iterations):
        final_best_model_params_file=os.path.join(base_dir,'best_model_params_%d.pkl'%(iteration))
        tf_networks['network_test'].load_model(final_best_model_params_file)    
        tf_networks['network_test'].reset_ewc_variables(session) # should not have ewc at final evaluation time  
        feature_maps=test_get_feature_maps(X,test_batch_size,feature_map_tensors,session=session,dataset=dataset)
        feature_map_list.append(feature_maps)
    return feature_map_list
def nan_test(network):
    params=network.get_all_model_params()
    arr_nan_test(params)
def arr_nan_test(arr):
    nan_list=[]
    inf_list=[]
    for k,v in arr.items():
        if not np.all(~np.isnan(v)):
            nan_list.append(k)
        if not np.all(~np.isinf(v)):
            nan_list.append(k)
    if len(nan_list)>0:
        print('nan list')
        print(nan_list)
    if len(inf_list)>0:
        print('inf list')
        print(inf_list)
    if len(nan_list)>0 or len(inf_list)>0:
        assert False
def print_mean(variable):
    for k,v in variable.items():
        print('%s:%e'%(k,np.mean(v)))
def print_max(variable):
    for k,v in variable.items():
        print('%s:%e'%(k,np.max(v)))
def abs_diff(v1,v2):
    diff_dict={}
    for k in v1.keys():
        diff_dict[k]=np.abs(v1[k]-v2[k])
    return diff_dict
def myfun():
    pass
