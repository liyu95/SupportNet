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
from nn_lib import *
from cifar100_utils import *
import matplotlib.pyplot as plt
import json
from pprint import pprint
from train_utils import *
np.random.seed(1997)
def build_graph_one_class_batch(hyper_params,fixed_params):
    tf.reset_default_graph()
    if 'random_seed' in fixed_params:
        tf.set_random_seed(fixed_params['random_seed'])
    tf_tensors={}
    tf_variables={}
    tf_networks={}
    tf_networks['train_network']=ResNet('ResNet32','train',num_outputs=fixed_params['total_num_classes'],name='ResNet32_train')
    tf_tensors['X_train']=tf_networks['train_network'].tf_tensors['input']
    tf_tensors['Y_train']=tf.placeholder(tf.int32,shape=[None])
    tf_tensors['Y_one_hot_train']=tf.one_hot(tf_tensors['Y_train'],fixed_params['total_num_classes'])
    tf_tensors['class_loss_train']=tf.reduce_mean(
               tf.nn.sigmoid_cross_entropy_with_logits(
               labels=tf_tensors['Y_one_hot_train'],logits=tf_networks['train_network'].tf_tensors['fc']))
    tf_tensors['regularization_loss_train']=hyper_params['beta']*tf_networks['train_network'].tf_tensors['l2_loss']
    tf_tensors['loss_train']=tf_tensors['class_loss_train']+tf_tensors['regularization_loss_train']
    tf_variables['lr']=tf.get_variable('lr',initializer=tf.constant(hyper_params['initial_lr'],dtype=tf.float32),trainable=False,dtype=tf.float32)
    tf_tensors['optimizer_train']=tf.train.MomentumOptimizer(learning_rate=tf_variables['lr'],momentum=0.9).minimize(tf_tensors['loss_train'])
#     tf_tensors['optimizer_train']=tf.train.AdamOptimizer(learning_rate=tf_variables['lr']).minimize(tf_tensors['loss_train'])
    indices_of_ranks_train=tf.nn.top_k(-tf_networks['train_network'].tf_tensors['fc'],k=fixed_params['total_num_classes'])[1]
    ranks_of_indices_train=tf.nn.top_k(-indices_of_ranks_train,k=fixed_params['total_num_classes'])[1]
    tf_tensors['ranks_of_indices_train']=ranks_of_indices_train
    indexing_matrix_train=tf.stack([tf.range(tf.shape(tf_tensors['Y_train'])[0]),tf_tensors['Y_train']],axis=1)
    ranks_of_groud_truth_class_train=fixed_params['total_num_classes']-1-tf.gather_nd(ranks_of_indices_train,indexing_matrix_train)
    tf_tensors['ranks_of_groud_truth_class_train']=ranks_of_groud_truth_class_train
    tf_tensors['top1_accuracy_train']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_train,1),tf.float32))
    tf_tensors['top5_accuracy_train']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_train,5),tf.float32))
    
    tf_networks['test_network']=ResNet('ResNet32','test',num_outputs=fixed_params['total_num_classes'],name='ResNet32_test')
    tf_tensors['X_test']=tf_networks['test_network'].tf_tensors['input']
    tf_tensors['Y_test']=tf.placeholder(tf.int32,shape=[None])
    tf_tensors['Y_one_hot_test']=tf.one_hot(tf_tensors['Y_test'],fixed_params['total_num_classes'])
    tf_tensors['class_loss_test']=tf.reduce_mean(
               tf.nn.sigmoid_cross_entropy_with_logits(
               labels=tf_tensors['Y_one_hot_test'],logits=tf_networks['test_network'].tf_tensors['fc']))
    tf_tensors['regularization_loss_test']=hyper_params['beta']*tf_networks['test_network'].tf_tensors['l2_loss']
    #print(tf_networks['test_network'].tf_tensors['l2_loss'])
    tf_tensors['loss_test']=tf_tensors['class_loss_test']+tf_tensors['regularization_loss_test']
    indices_of_ranks_test=tf.nn.top_k(-tf_networks['test_network'].tf_tensors['fc'],k=fixed_params['total_num_classes'])[1]
    ranks_of_indices_test=tf.nn.top_k(-indices_of_ranks_test,k=fixed_params['total_num_classes'])[1]
    tf_tensors['ranks_of_indices_test']=ranks_of_indices_test
    indexing_matrix_test=tf.stack([tf.range(tf.shape(tf_tensors['Y_test'])[0]),tf_tensors['Y_test']],axis=1)
    ranks_of_groud_truth_class_test=fixed_params['total_num_classes']-1-tf.gather_nd(ranks_of_indices_test,indexing_matrix_test)
    tf_tensors['ranks_of_groud_truth_class_test']=ranks_of_groud_truth_class_test
    tf_tensors['top1_accuracy_test']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_test,1),tf.float32))
    tf_tensors['top5_accuracy_test']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_test,5),tf.float32))    
    
    local_step_var=tf.get_variable('local_step',initializer=tf.constant(0),trainable=False)
    tf_variables['local_step_var']=local_step_var

    global_step_var=tf.get_variable('global_step',initializer=tf.constant(0),trainable=False)
    tf_variables['global_step_var']=global_step_var
    
    loss_train_var=tf.get_variable('loss_train',initializer=tf.constant(np.Infinity,dtype=tf.float32),trainable=False)
    tf_variables['loss_train_var']=loss_train_var
    
    loss_test_var=tf.get_variable('loss_test',initializer=tf.constant(np.Infinity,dtype=tf.float32),trainable=False)
    tf_variables['loss_test_var']=loss_test_var
    
    best_loss_test_var=tf.get_variable('best_loss_test',initializer=tf.constant(np.Infinity,dtype=tf.float32),trainable=False)
    tf_variables['best_loss_test_var']=best_loss_test_var
    
    best_loss_test_global_step_var=tf.get_variable('best_loss_test_global_step',initializer=tf.constant(0),trainable=False)
    tf_variables['best_loss_test_global_step_var']=best_loss_test_global_step_var
    
    top1_accuracy_train_var=tf.get_variable('top1_accuracy_train',initializer=tf.constant(0,dtype=tf.float32),trainable=False)
    tf_variables['top1_accuracy_train_var']=top1_accuracy_train_var
    
    top1_accuracy_test_var=tf.get_variable('top1_accuracy_test',initializer=tf.constant(0,dtype=tf.float32),trainable=False)
    tf_variables['top1_accuracy_test_var']=top1_accuracy_test_var
    
    best_top1_accuracy_test_var=tf.get_variable('best_top1_accuracy_test',initializer=tf.constant(0,dtype=tf.float32),trainable=False)
    tf_variables['best_top1_accuracy_test_var']=best_top1_accuracy_test_var

    best_top1_accuracy_test_global_step_var=tf.get_variable('best_top1_accuracy_test_global_step',initializer=tf.constant(0),trainable=False)
    tf_variables['best_top1_accuracy_test_global_step_var']=best_top1_accuracy_test_global_step_var
    
    top5_accuracy_train_var=tf.get_variable('top5_accuracy_train',initializer=tf.constant(0,dtype=tf.float32),trainable=False)
    tf_variables['top5_accuracy_train_var']=top5_accuracy_train_var
    
    top5_accuracy_test_var=tf.get_variable('top5_accuracy_test',initializer=tf.constant(0,dtype=tf.float32),trainable=False)
    tf_variables['top5_accuracy_test_var']=top5_accuracy_test_var
    
    best_top5_accuracy_test_var=tf.get_variable('best_top5_accuracy_test',initializer=tf.constant(0,dtype=tf.float32),trainable=False)
    tf_variables['best_top5_accuracy_test_var']=best_top5_accuracy_test_var

    best_top5_accuracy_test_global_step_var=tf.get_variable('best_top5_accuracy_test_global_step',initializer=tf.constant(0),trainable=False)
    tf_variables['best_top5_accuracy_test_global_step_var']=best_top5_accuracy_test_global_step_var
    
    return tf_tensors,tf_variables,tf_networks
    
    
def fit_one_class_batch(tf_tensors,tf_variables,tf_networks,fixed_params,hyper_params,data_dict,session,resume=False,
        save_session=True,save_session_freq=1,save_params=True,evaluation_metric='top1_accuracy',save_history=True,
        num_epochs=40,verbose=2,print_freq=1,override_warning=True):
    tf_log_var_names=['local_step','global_step',
                'loss_train','loss_test','best_loss_test','best_loss_test_global_step',
                'top1_accuracy_train','top1_accuracy_test','best_top1_accuracy_test','best_top1_accuracy_test_global_step',
                'top5_accuracy_train','top5_accuracy_test','best_top5_accuracy_test','best_top5_accuracy_test_global_step']
    if verbose>=2:
        pprint(fixed_params)
        pprint(hyper_params)
    tensorboard_dir=os.path.join(fixed_params['base_dir'],'tensorboard_dir')
    checkpoint_dir=os.path.join(fixed_params['base_dir'],'checkpoint_dir')
    hyper_params_file=os.path.join(fixed_params['base_dir'],'hyper_params.json')
    fixed_params_file=os.path.join(fixed_params['base_dir'],'fixed_params.json')
    best_model_eval_metric_file=os.path.join(fixed_params['base_dir'],'best_model_eval_metric.json')
    best_model_params_file=os.path.join(fixed_params['base_dir'],'best_model_params.pkl')
    history_file=os.path.join(fixed_params['base_dir'],'history.json')
    latest_ckpt=tf.train.latest_checkpoint(checkpoint_dir)
    saver=tf.train.Saver(max_to_keep=None)
    if resume and latest_ckpt:
        saver.restore(session,latest_ckpt)
        if save_history:
            history_dict=json.load(open(history_file))
        train_writer=tf.summary.FileWriter(os.path.join(tensorboard_dir,'train'),tf.get_default_graph())
        test_writer=tf.summary.FileWriter(os.path.join(tensorboard_dir,'test'))
    else:
        if os.path.exists(fixed_params['base_dir']):
            if override_warning:
                choice=input(fixed_params['base_dir']+' already exists, override?')
            else:
                choice='y'
            if choice=='y':
                shutil.rmtree(fixed_params['base_dir'])
                os.makedirs(tensorboard_dir)
                os.makedirs(checkpoint_dir)
            else:
                print('cancelled')
                return
        else:
            os.makedirs(tensorboard_dir)
            os.makedirs(checkpoint_dir)
        train_writer=tf.summary.FileWriter(os.path.join(tensorboard_dir,'train'),tf.get_default_graph())
        test_writer=tf.summary.FileWriter(os.path.join(tensorboard_dir,'test')) #try
        with open(hyper_params_file,'w') as f_hyper, open(fixed_params_file,'w') as f_fixed:
            json.dump(hyper_params,f_hyper,indent=4)
            json.dump(fixed_params,f_fixed,indent=4)
            
        init=tf.global_variables_initializer()
        if save_history:
            history_dict={n:[] for n in tf_log_var_names}
        session.run(init)
    
    tf_log_var_val_list=session.run([tf_variables[v+'_var'] for v in tf_log_var_names])
    tf_log_var_val=dict(zip(tf_log_var_names,tf_log_var_val_list))
    train_idx=np.array([i in fixed_params['using_classes'] for i in data_dict['Y_train']])
    test_idx=np.array([i in fixed_params['using_classes'] for i in data_dict['Y_test']])
    #pre-train evaluation
    if tf_log_var_val['global_step']==0:
        #evaluation on trainset
        evaluation_tensors=dict(loss=tf_tensors['loss_train'],
                                class_loss=tf_tensors['class_loss_train'],
                                regularization_loss=tf_tensors['top1_accuracy_train'],
                                top1_accuracy=tf_tensors['top1_accuracy_train'],
                                top5_accuracy=tf_tensors['top5_accuracy_train'],
                                X=tf_tensors['X_train'],
                                Y=tf_tensors['Y_train'])
        evaluation_vals=test_accuracy_evaluation_plain(data_dict['X_train'][train_idx],
                                                       data_dict['Y_train'][train_idx],
                                                       hyper_params['test_batch_size'],
                                                       evaluation_tensors,session)
        tf_log_var_val.update({k+'_train':evaluation_vals[k] for k in evaluation_vals})
        #evaluation on testset
        evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                class_loss=tf_tensors['class_loss_test'],
                                regularization_loss=tf_tensors['top1_accuracy_test'],
                                top1_accuracy=tf_tensors['top1_accuracy_test'],
                                top5_accuracy=tf_tensors['top5_accuracy_test'],
                                X=tf_tensors['X_test'],
                                Y=tf_tensors['Y_test'])
        evaluation_vals=test_accuracy_evaluation_plain(data_dict['X_test'][test_idx],
                                                       data_dict['Y_test'][test_idx],
                                                       hyper_params['test_batch_size'],
                                                       evaluation_tensors,session)
        tf_log_var_val.update({k+'_test':evaluation_vals[k] for k in evaluation_vals})        
    
            
        if verbose>=2 and tf_log_var_val['global_step']%print_freq==0:
            print('Pretrain evaluation')
            print("\tTrain loss: %f"%tf_log_var_val['loss_train'])
            print("\tTrain class loss: %f"%tf_log_var_val['class_loss_train'])
            print("\tTrain reg loss: %f"%tf_log_var_val['regularization_loss_train'])
            print("\tValidation loss: %f"%(tf_log_var_val['loss_test']))
            print("\tValidation class loss: %f"%tf_log_var_val['class_loss_test'])
            print("\tValidation reg loss: %f"%tf_log_var_val['regularization_loss_test'])
            print("\tTop1 train accuracy: %f"%(tf_log_var_val['top1_accuracy_train']))
            print("\tTop5 train accuracy: %f"%(tf_log_var_val['top5_accuracy_train']))
            print("\tTop1 validation accuracy: %f"%(tf_log_var_val['top1_accuracy_test']))
            print("\tTop5 validation accuracy: %f"%(tf_log_var_val['top5_accuracy_test']))

    for epoch in range(tf_log_var_val['global_step'],num_epochs):
        if tf_log_var_val['global_step'] in hyper_params['lr_reduction_epoch']:
            new_lr=session.run(tf_variables['lr'].assign(tf_variables['lr']/hyper_params['lr_reduction_rate']))
            print('lr reduced to %f'%new_lr)
            
        tf_log_var_val['loss_train']=0
        tf_log_var_val['class_loss_train']=0
        tf_log_var_val['regularization_loss_train']=0
        tf_log_var_val['top1_accuracy_train']=0
        tf_log_var_val['top5_accuracy_train']=0
        num_batches_train=int((np.sum(train_idx)/hyper_params['train_batch_size']))
        for X_minibatch,Y_minibatch in iterate_minibatches(data_dict['X_train'][train_idx],data_dict['Y_train'][train_idx],hyper_params['train_batch_size'],shuffle=True,augment=True):
            _,local_loss,local_class_loss,local_regularization_loss,local_top1_accuracy,local_top5_accuracy\
            =session.run([tf_tensors['optimizer_train'],tf_tensors['loss_train'],tf_tensors['class_loss_train'],tf_tensors['regularization_loss_train'],tf_tensors['top1_accuracy_train'],tf_tensors['top5_accuracy_train']],
                                      feed_dict={tf_tensors['X_train']:X_minibatch,
                                                 tf_tensors['Y_train']:Y_minibatch})
            tf_log_var_val['local_step']+=1
            tf_log_var_val['loss_train']+=local_loss/num_batches_train
            tf_log_var_val['class_loss_train']+=local_class_loss/num_batches_train
            tf_log_var_val['regularization_loss_train']+=local_regularization_loss/num_batches_train
            tf_log_var_val['top1_accuracy_train']+=local_top1_accuracy/num_batches_train
            tf_log_var_val['top5_accuracy_train']+=local_top5_accuracy/num_batches_train
            
        tf_log_var_val['global_step']+=1
        
        model_params=tf_networks['train_network'].get_all_model_params(session)
        tf_networks['test_network'].set_model_params(model_params,session)
        
        #evaluation on testset
        evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                class_loss=tf_tensors['class_loss_test'],
                                regularization_loss=tf_tensors['top1_accuracy_test'],
                                top1_accuracy=tf_tensors['top1_accuracy_test'],
                                top5_accuracy=tf_tensors['top5_accuracy_test'],
                                X=tf_tensors['X_test'],
                                Y=tf_tensors['Y_test'])
        evaluation_vals=test_accuracy_evaluation_plain(data_dict['X_test'][test_idx],
                                                       data_dict['Y_test'][test_idx],
                                                       hyper_params['test_batch_size'],
                                                       evaluation_tensors,session)
        tf_log_var_val.update({k+'_test':evaluation_vals[k] for k in evaluation_vals}) 
            
        if evaluation_metric=='top1_accuracy':
            best_metric_val=tf_log_var_val['best_top1_accuracy_test']
        elif evaluation_metric=='top5_accuracy':
            best_metric_val=tf_log_var_val['est_top5_accuracy_test']
        elif evaluation_metric=='validation_loss':
            best_metric_val=-best_tf_log_var_val['loss_test']
        else:
            raise ValueError('Unknown evaluation metric: '+evaluation_metric)
            
        if verbose>=2 and tf_log_var_val['global_step']%print_freq==0:
            print('Epoch %d'%(tf_log_var_val['global_step']))
            print("\tTrain loss: %f"%tf_log_var_val['loss_train'])
            print("\tTrain class loss: %f"%tf_log_var_val['class_loss_train'])
            print("\tTrain reg loss: %f"%tf_log_var_val['regularization_loss_train'])
            print("\tValidation loss: %f"%(tf_log_var_val['loss_test']))
            print("\tValidation class loss: %f"%tf_log_var_val['class_loss_test'])
            print("\tValidation reg loss: %f"%tf_log_var_val['regularization_loss_test'])
            print("\tTop1 train accuracy: %f"%(tf_log_var_val['top1_accuracy_train']))
            print("\tTop5 train accuracy: %f"%(tf_log_var_val['top5_accuracy_train']))
            print("\tTop1 validation accuracy: %f"%(tf_log_var_val['top1_accuracy_test']))
            print("\tTop5 validation accuracy: %f"%(tf_log_var_val['top5_accuracy_test']))

        if tf_log_var_val['loss_test']<tf_log_var_val['best_loss_test']:
            tf_log_var_val['best_loss_test']=tf_log_var_val['loss_test']
            tf_log_var_val['best_loss_test_global_step']=tf_log_var_val['global_step']
        if tf_log_var_val['top1_accuracy_test']>tf_log_var_val['best_top1_accuracy_test']:
            tf_log_var_val['best_top1_accuracy_test']=tf_log_var_val['top1_accuracy_test']
            tf_log_var_val['best_top1_accuracy_test_global_step']=tf_log_var_val['global_step']
        if tf_log_var_val['top5_accuracy_test']>tf_log_var_val['best_top5_accuracy_test']:
                tf_log_var_val['best_top5_accuracy_test']=tf_log_var_val['top5_accuracy_test']
                tf_log_var_val['best_top5_accuracy_test_global_step']=tf_log_var_val['global_step'] 
        
        tf_log_var_names_assign_ops=[tf_variables[v+'_var'].assign(tf_log_var_val[v]) for v in tf_log_var_names]
        session.run(tf_log_var_names_assign_ops)
        for v in tf_log_var_names:
            value=tf_log_var_val[v]
            if isinstance(value,(np.int32,np.int64)):
                history_dict[v].append(int(value))
            elif isinstance(value,(np.float32,np.float64)):
                history_dict[v].append(float(value))
            else:
                history_dict[v].append(value)
        if save_history:
            with open(history_file,'w') as f:
                json.dump(history_dict,f,indent=4)
        if save_session and tf_log_var_val['global_step']%save_session_freq==0:
            saver.save(session,os.path.join(checkpoint_dir,'my_model'))
            if verbose>=2:
                print('written global step %d'%(tf_log_var_val['global_step']))
            
        if evaluation_metric=='top1_accuracy':
            new_metric_val=tf_log_var_val['top1_accuracy_test']
        elif evaluation_metric=='top5_accuracy':
            new_metric_val=tf_log_var_val['top5_accuracy_test']
        elif evaluation_metric=='validation_loss':
            new_metric_val=-tf_log_var_val['loss_test']

        if new_metric_val>best_metric_val:
            print('saving model parameters...')
            tf_networks['train_network'].save_model(best_model_params_file,session)
            for k in tf_log_var_val:
#                 print(k+' '+str(tf_log_var_val[k])+' '+str(type(tf_log_var_val[k])))
                if isinstance(tf_log_var_val[k],np.int64):
                    tf_log_var_val[k]=int(tf_log_var_val[k])
                if isinstance(tf_log_var_val[k],np.int32):
                    tf_log_var_val[k]=int(tf_log_var_val[k])
                if isinstance(tf_log_var_val[k],np.float32):
                    tf_log_var_val[k]=float(tf_log_var_val[k])
                    
            with open(best_model_eval_metric_file,'w') as f:
                json.dump(tf_log_var_val,f,indent=4)
    if verbose>=1:
        print('Final best global step: %d'%tf_log_var_val['global_step'])
        print('Final best top1 validation accuracy: %f'%tf_log_var_val['best_top1_accuracy_test'])
        print('Final best top5 validation accuracy: %f'%tf_log_var_val['best_top5_accuracy_test'])
        print('Final best validation loss: %f'%tf_log_var_val['best_loss_test'])
    train_writer.close()
    test_writer.close()
#################################################################################################################################
def build_graph_multiple_class_batches(hyper_params,fixed_params):
    tf.reset_default_graph()
    if 'random_seed' in fixed_params:
        tf.set_random_seed(fixed_params['random_seed'])
    tf_tensors={}
    tf_variables={}
    tf_networks={}
    #change to 'test'
    tf_networks['train_network']=ResNet('ResNet32','train',num_outputs=fixed_params['total_num_classes'],name='ResNet32_train')
    tf_tensors['X_train']=tf_networks['train_network'].tf_tensors['input']
    tf_tensors['Y_train']=tf.placeholder(tf.int32,shape=[None])
    tf_tensors['Y_one_hot_train']=tf.one_hot(tf_tensors['Y_train'],fixed_params['total_num_classes'])
    tf_tensors['class_loss_train']=tf.reduce_mean(
               tf.nn.sigmoid_cross_entropy_with_logits(
               labels=tf_tensors['Y_one_hot_train'],logits=tf_networks['train_network'].tf_tensors['fc']))
    tf_tensors['regularization_loss_train']=hyper_params['beta']*tf_networks['train_network'].tf_tensors['l2_loss']
    tf_tensors['loss_train']=tf_tensors['class_loss_train']+tf_tensors['regularization_loss_train']
    tf_variables['lr']=tf.get_variable('lr',initializer=tf.constant(hyper_params['initial_lr'],dtype=tf.float32),trainable=False,dtype=tf.float32)
    tf_tensors['optimizer_train']=tf.train.MomentumOptimizer(learning_rate=tf_variables['lr'],momentum=0.9).minimize(tf_tensors['loss_train'])
#     tf_tensors['optimizer_train']=tf.train.AdamOptimizer(learning_rate=tf_variables['lr']).minimize(tf_tensors['loss_train'])
    indices_of_ranks_train=tf.nn.top_k(-tf_networks['train_network'].tf_tensors['fc'],k=fixed_params['total_num_classes'])[1]
    ranks_of_indices_train=tf.nn.top_k(-indices_of_ranks_train,k=fixed_params['total_num_classes'])[1]
    tf_tensors['ranks_of_indices_train']=ranks_of_indices_train
    indexing_matrix_train=tf.stack([tf.range(tf.shape(tf_tensors['Y_train'])[0]),tf_tensors['Y_train']],axis=1)
    ranks_of_groud_truth_class_train=fixed_params['total_num_classes']-1-tf.gather_nd(ranks_of_indices_train,indexing_matrix_train)
    tf_tensors['ranks_of_groud_truth_class_train']=ranks_of_groud_truth_class_train
    tf_tensors['top1_accuracy_train']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_train,1),tf.float32))
    tf_tensors['top5_accuracy_train']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_train,5),tf.float32))
    
    tf_networks['test_network']=ResNet('ResNet32','test',num_outputs=fixed_params['total_num_classes'],name='ResNet32_test')
    tf_tensors['X_test']=tf_networks['test_network'].tf_tensors['input']
    tf_tensors['Y_test']=tf.placeholder(tf.int32,shape=[None])
    tf_tensors['Y_one_hot_test']=tf.one_hot(tf_tensors['Y_test'],fixed_params['total_num_classes'])
    tf_tensors['class_loss_test']=tf.reduce_mean(
               tf.nn.sigmoid_cross_entropy_with_logits(
               labels=tf_tensors['Y_one_hot_test'],logits=tf_networks['test_network'].tf_tensors['fc']))
    tf_tensors['regularization_loss_test']=hyper_params['beta']*tf_networks['test_network'].tf_tensors['l2_loss']
    #print(tf_networks['test_network'].tf_tensors['l2_loss'])
    tf_tensors['loss_test']=tf_tensors['class_loss_test']+tf_tensors['regularization_loss_test']
    indices_of_ranks_test=tf.nn.top_k(-tf_networks['test_network'].tf_tensors['fc'],k=fixed_params['total_num_classes'])[1]
    ranks_of_indices_test=tf.nn.top_k(-indices_of_ranks_test,k=fixed_params['total_num_classes'])[1]
    tf_tensors['ranks_of_indices_test']=ranks_of_indices_test
    indexing_matrix_test=tf.stack([tf.range(tf.shape(tf_tensors['Y_test'])[0]),tf_tensors['Y_test']],axis=1)
    ranks_of_groud_truth_class_test=fixed_params['total_num_classes']-1-tf.gather_nd(ranks_of_indices_test,indexing_matrix_test)
    tf_tensors['ranks_of_groud_truth_class_test']=ranks_of_groud_truth_class_test
    tf_tensors['top1_accuracy_test']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_test,1),tf.float32))
    tf_tensors['top5_accuracy_test']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_test,5),tf.float32))    
    
    iteration_step_var=tf.get_variable('iteration_step',initializer=tf.constant(0),trainable=False)
    tf_variables['iteration_step_var']=iteration_step_var
    
    global_step_var=tf.get_variable('global_step',initializer=tf.constant(0),trainable=False)
    tf_variables['global_step_var']=global_step_var
    
    local_step_var=tf.get_variable('local_step',initializer=tf.constant(0),trainable=False)
    tf_variables['local_step_var']=local_step_var

    loss_train_var=tf.get_variable('loss_train',initializer=tf.constant(np.Infinity,dtype=tf.float32),trainable=False)
    tf_variables['loss_train_var']=loss_train_var
    
    loss_test_var=tf.get_variable('loss_test',initializer=tf.constant(np.Infinity,dtype=tf.float32),trainable=False)
    tf_variables['loss_test_var']=loss_test_var
    
    best_loss_test_var=tf.get_variable('best_loss_test',initializer=tf.constant(np.Infinity,dtype=tf.float32),trainable=False)
    tf_variables['best_loss_test_var']=best_loss_test_var
    
    best_loss_test_global_step_var=tf.get_variable('best_loss_test_global_step',initializer=tf.constant(0),trainable=False)
    tf_variables['best_loss_test_epoch_var']=best_loss_test_global_step_var
    
    top1_accuracy_train_var=tf.get_variable('top1_accuracy_train',initializer=tf.constant(0,dtype=tf.float32),trainable=False)
    tf_variables['top1_accuracy_train_var']=top1_accuracy_train_var
    
    top1_accuracy_test_var=tf.get_variable('top1_accuracy_test',initializer=tf.constant(0,dtype=tf.float32),trainable=False)
    tf_variables['top1_accuracy_test_var']=top1_accuracy_test_var
    
    best_top1_accuracy_test_var=tf.get_variable('best_top1_accuracy_test',initializer=tf.constant(0,dtype=tf.float32),trainable=False)
    tf_variables['best_top1_accuracy_test_var']=best_top1_accuracy_test_var

    best_top1_accuracy_test_global_step_var=tf.get_variable('best_top1_accuracy_test_global_step',initializer=tf.constant(0),trainable=False)
    tf_variables['best_top1_accuracy_test_epoch_var']=best_top1_accuracy_test_global_step_var
    
    top5_accuracy_train_var=tf.get_variable('top5_accuracy_train',initializer=tf.constant(0,dtype=tf.float32),trainable=False)
    tf_variables['top5_accuracy_train_var']=top5_accuracy_train_var
    
    top5_accuracy_test_var=tf.get_variable('top5_accuracy_test',initializer=tf.constant(0,dtype=tf.float32),trainable=False)
    tf_variables['top5_accuracy_test_var']=top5_accuracy_test_var
    
    best_top5_accuracy_test_var=tf.get_variable('best_top5_accuracy_test',initializer=tf.constant(0,dtype=tf.float32),trainable=False)
    tf_variables['best_top5_accuracy_test_var']=best_top5_accuracy_test_var

    best_top5_accuracy_test_global_step_var=tf.get_variable('best_top5_accuracy_test_global_step',initializer=tf.constant(0),trainable=False)
    tf_variables['best_top5_accuracy_test_epoch_var']=best_top5_accuracy_test_global_step_var
    
    return tf_tensors,tf_variables,tf_networks    
def fit_multiple_class_batches(tf_tensors,tf_variables,tf_networks,fixed_params,hyper_params,data_dict_total,session,resume=False,
        save_session=True,save_session_freq=1,save_params=True,evaluation_metric='top1_accuracy',save_history=True,
        num_epochs=40,num_iterations=10,verbose=2,print_freq=1,pretrain_evaluation=1,override_warning=True):
    tf_log_var_names=['local_step','global_step','iteration_step',
                'loss_train','loss_test','best_loss_test','best_loss_test_epoch',
                'top1_accuracy_train','top1_accuracy_test','best_top1_accuracy_test','best_top1_accuracy_test_epoch',
                'top5_accuracy_train','top5_accuracy_test','best_top5_accuracy_test','best_top5_accuracy_test_epoch']
    if verbose>=2:
        pprint(fixed_params)
        pprint(hyper_params)
        
    #determine class batch numbers
    max_num_iterations=np.ceil(fixed_params['total_num_classes']/fixed_params['class_batch_size'])
    if num_iterations>max_num_iterations:
        num_iterations=max_num_iterations
        print('num_iterations exceeds maximum allowed number, reset to '+str(max_num_iterations))
        
    # get output file names
    tensorboard_dir=os.path.join(fixed_params['base_dir'],'tensorboard_dir')
    checkpoint_dir=os.path.join(fixed_params['base_dir'],'checkpoint_dir')
    hyper_params_file=os.path.join(fixed_params['base_dir'],'hyper_params.json')
    fixed_params_file=os.path.join(fixed_params['base_dir'],'fixed_params.json')
    history_file=os.path.join(fixed_params['base_dir'],'history.json')
    class_ord_file=os.path.join(fixed_params['base_dir'],'class_ord.json')
    
    #saving,restoring and initializing
    latest_ckpt=tf.train.latest_checkpoint(checkpoint_dir)
    saver=tf.train.Saver(max_to_keep=None)
    if resume and latest_ckpt:
        saver.restore(session,latest_ckpt)
        if save_history:
            with open(history_file) as f:
                history=json.load(f)
        with open(class_ord_file) as f_class_ord:
            class_ord=json.load(f_class_ord)
        train_writer=tf.summary.FileWriter(os.path.join(tensorboard_dir,'train'),tf.get_default_graph())
        test_writer=tf.summary.FileWriter(os.path.join(tensorboard_dir,'test'))
    else:
        if os.path.exists(fixed_params['base_dir']):
            if override_warning:
                choice=input(fixed_params['base_dir']+' already exists, override?')
            else:
                choice='y'
            if choice=='y':
                shutil.rmtree(fixed_params['base_dir'])
                os.makedirs(tensorboard_dir)
                os.makedirs(checkpoint_dir)
            else:
                print('cancelled')
                return
        else:
            os.makedirs(tensorboard_dir)
            os.makedirs(checkpoint_dir)
        train_writer=tf.summary.FileWriter(os.path.join(tensorboard_dir,'train'),tf.get_default_graph())
        test_writer=tf.summary.FileWriter(os.path.join(tensorboard_dir,'test')) #try
        with open(hyper_params_file,'w') as f_hyper, open(fixed_params_file,'w') as f_fixed:
            json.dump(hyper_params,f_hyper,indent=4)
            json.dump(fixed_params,f_fixed,indent=4)
            
        init=tf.global_variables_initializer()
        session.run(init)
        tf_networks['test_network'].set_model_params(tf_networks['train_network'].get_all_model_params(session),session)
        # init history
        if save_history:
            history=[{n:[] for n in tf_log_var_names} for _ in range(num_iterations)]
        #init class_ord
        class_ord=np.arange(fixed_params['total_num_classes'])
        np.random.shuffle(class_ord)
        class_ord=[int(i) for i in class_ord]
        with open(class_ord_file,'w') as f_class_ord:
            json.dump(class_ord,f_class_ord,indent=4)
    #restore log variables
    tf_log_var_val_list=session.run([tf_variables[v+'_var'] for v in tf_log_var_names])
    tf_log_var_val=dict(zip(tf_log_var_names,tf_log_var_val_list))

           
    #pre-train evaluation on all class
    if tf_log_var_val['global_step']==0 and tf_log_var_val['iteration_step']==0 and pretrain_evaluation>=2:
        #evaluation on trainset
        evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                class_loss=tf_tensors['class_loss_test'],
                                regularization_loss=tf_tensors['top1_accuracy_test'],
                                top1_accuracy=tf_tensors['top1_accuracy_test'],
                                top5_accuracy=tf_tensors['top5_accuracy_test'],
                                X=tf_tensors['X_test'],
                                Y=tf_tensors['Y_test'])
        evaluation_vals=test_accuracy_evaluation_plain(data_dict_total['X_train'],
                                                       data_dict_total['Y_train'],
                                                       hyper_params['test_batch_size'],
                                                       evaluation_tensors,session)
        tf_log_var_val.update({k+'_train':evaluation_vals[k] for k in evaluation_vals})
        #evaluation on testset
        evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                class_loss=tf_tensors['class_loss_test'],
                                regularization_loss=tf_tensors['top1_accuracy_test'],
                                top1_accuracy=tf_tensors['top1_accuracy_test'],
                                top5_accuracy=tf_tensors['top5_accuracy_test'],
                                X=tf_tensors['X_test'],
                                Y=tf_tensors['Y_test'])
        evaluation_vals=test_accuracy_evaluation_plain(data_dict_total['X_test'],
                                                       data_dict_total['Y_test'],
                                                       hyper_params['test_batch_size'],
                                                       evaluation_tensors,session)
        tf_log_var_val.update({k+'_test':evaluation_vals[k] for k in evaluation_vals})        
    
            
        if verbose>=2:
            print_var_list=['loss_train','class_loss_train','regularization_loss_train',
                           'loss_test','class_loss_test','regularization_loss_test',
                           'top1_accuracy_train','top5_accuracy_train','top1_accuracy_test','top5_accuracy_test']
            print_eval_metric(tf_log_var_val,print_var_list,'Pretrain evaluation on all classes')  

    for iteration in range(tf_log_var_val['iteration_step'],num_iterations):
        #train and test file for each class_batch
        using_classes=class_ord[iteration*fixed_params['class_batch_size']:(iteration+1)*fixed_params['class_batch_size']]
        train_idx=np.array([i in using_classes for i in data_dict_total['Y_train']])
        test_idx=np.array([i in using_classes for i in data_dict_total['Y_test']])
        data_dict=dict(X_train=data_dict_total['X_train'][train_idx],
                       Y_train=data_dict_total['Y_train'][train_idx],
                       X_test=data_dict_total['X_test'][test_idx],
                       Y_test=data_dict_total['Y_test'][test_idx])
        
        #set filename for this iteration
        best_model_eval_metric_file=os.path.join(fixed_params['base_dir'],'best_model_eval_metric_%d.json'%iteration)  
        best_model_params_file=os.path.join(fixed_params['base_dir'],'best_model_params_%d.pkl'%iteration) #save best params after every iteration
        
        #restore lr at start of iteration
        if tf_log_var_val['global_step']%num_epochs==0:
            session.run(tf_variables['lr'].assign(hyper_params['initial_lr']))
            
            reset_var_list=['loss_train','loss_test','best_loss_test','best_loss_test_epoch',
                            'top1_accuracy_train','top1_accuracy_test','best_top1_accuracy_test','best_top1_accuracy_test_epoch',
                            'top5_accuracy_train','top5_accuracy_test','best_top5_accuracy_test','best_top5_accuracy_test_epoch']
            reset_tf_log_var_val(tf_log_var_val,reset_var_list)
        print('===========Iteration %d============='%(iteration+1))
        print('Using classes %r'%using_classes)
        for epoch in range(tf_log_var_val['global_step']%num_epochs,num_epochs):
            if epoch in hyper_params['lr_reduction_epoch']:
                new_lr=session.run(tf_variables['lr'].assign(tf_variables['lr']/hyper_params['lr_reduction_rate']))
                print('lr reduced to %f'%new_lr)
            #class batch pretrain evaluation
            if epoch==0 and pretrain_evaluation>=1:
                #evaluation on trainset
                evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                        class_loss=tf_tensors['class_loss_test'],
                                        regularization_loss=tf_tensors['top1_accuracy_test'],
                                        top1_accuracy=tf_tensors['top1_accuracy_test'],
                                        top5_accuracy=tf_tensors['top5_accuracy_test'],
                                        X=tf_tensors['X_test'],
                                        Y=tf_tensors['Y_test'])
                evaluation_vals=test_accuracy_evaluation_plain(data_dict['X_train'],
                                                               data_dict['Y_train'],
                                                               hyper_params['test_batch_size'],
                                                               evaluation_tensors,session)
                tf_log_var_val.update({k+'_train':evaluation_vals[k] for k in evaluation_vals})
                #evaluation on testset
                evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                        class_loss=tf_tensors['class_loss_test'],
                                        regularization_loss=tf_tensors['regularization_loss_test'],
                                        top1_accuracy=tf_tensors['top1_accuracy_test'],
                                        top5_accuracy=tf_tensors['top5_accuracy_test'],
                                        X=tf_tensors['X_test'],
                                        Y=tf_tensors['Y_test'])
                evaluation_vals=test_accuracy_evaluation_plain(data_dict['X_test'],
                                                               data_dict['Y_test'],
                                                               hyper_params['test_batch_size'],
                                                               evaluation_tensors,session)
                tf_log_var_val.update({k+'_test':evaluation_vals[k] for k in evaluation_vals})  
                
                if verbose>=2:
                    print_var_list=['loss_train','class_loss_train','regularization_loss_train',
                                   'loss_test','class_loss_test','regularization_loss_test',
                                   'top1_accuracy_train','top5_accuracy_train','top1_accuracy_test','top5_accuracy_test']
                    print_eval_metric(tf_log_var_val,print_var_list,'Class batch pretrain evaluation')                
            #training
            tf_log_var_val['loss_train']=0
            tf_log_var_val['class_loss_train']=0
            tf_log_var_val['regularization_loss_train']=0
            tf_log_var_val['top1_accuracy_train']=0
            tf_log_var_val['top5_accuracy_train']=0
            num_batches_train=int((len(data_dict['X_train'])/hyper_params['train_batch_size']))
            for X_minibatch,Y_minibatch in iterate_minibatches(data_dict['X_train'],data_dict['Y_train'],hyper_params['train_batch_size'],shuffle=True,augment=True):
                _,local_loss,local_class_loss,local_regularization_loss,local_top1_accuracy,local_top5_accuracy\
                =session.run([tf_tensors['optimizer_train'],tf_tensors['loss_train'],tf_tensors['class_loss_train'],tf_tensors['regularization_loss_train'],tf_tensors['top1_accuracy_train'],tf_tensors['top5_accuracy_train']],
                                          feed_dict={tf_tensors['X_train']:X_minibatch,
                                                     tf_tensors['Y_train']:Y_minibatch})
                tf_log_var_val['local_step']+=1
                tf_log_var_val['loss_train']+=local_loss/num_batches_train
                tf_log_var_val['class_loss_train']+=local_class_loss/num_batches_train
                tf_log_var_val['regularization_loss_train']+=local_regularization_loss/num_batches_train
                tf_log_var_val['top1_accuracy_train']+=local_top1_accuracy/num_batches_train
                tf_log_var_val['top5_accuracy_train']+=local_top5_accuracy/num_batches_train

            #get model parameters from train net
            model_params=tf_networks['train_network'].get_all_model_params(session)
            tf_networks['test_network'].set_model_params(model_params,session)

            #evaluation on testset
            evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                    class_loss=tf_tensors['class_loss_test'],
                                    regularization_loss=tf_tensors['regularization_loss_test'],
                                    top1_accuracy=tf_tensors['top1_accuracy_test'],
                                    top5_accuracy=tf_tensors['top5_accuracy_test'],
                                    X=tf_tensors['X_test'],
                                    Y=tf_tensors['Y_test'])
            evaluation_vals=test_accuracy_evaluation_plain(data_dict['X_test'],
                                                           data_dict['Y_test'],
                                                           hyper_params['test_batch_size'],
                                                           evaluation_tensors,session)
            tf_log_var_val.update({k+'_test':evaluation_vals[k] for k in evaluation_vals}) 
            
            tf_log_var_val['global_step']+=1
            
            #last epoch
            if epoch==num_epochs-1:
                tf_log_var_val['iteration_step']+=1
            
            if evaluation_metric=='top1_accuracy':
                best_metric_val=tf_log_var_val['best_top1_accuracy_test']
            elif evaluation_metric=='top5_accuracy':
                best_metric_val=tf_log_var_val['est_top5_accuracy_test']
            elif evaluation_metric=='validation_loss':
                best_metric_val=-best_tf_log_var_val['loss_test']
            else:
                raise ValueError('Unknown evaluation metric: '+evaluation_metric)

            if verbose>=2 and epoch%print_freq==0:
                
                print_var_list=['loss_train','class_loss_train','regularization_loss_train',
                               'loss_test','class_loss_test','regularization_loss_test',
                               'top1_accuracy_train','top5_accuracy_train','top1_accuracy_test','top5_accuracy_test']                
                print_eval_metric(tf_log_var_val,print_var_list,'Epoch %d'%(epoch+1))

            if tf_log_var_val['loss_test']<tf_log_var_val['best_loss_test']:
                tf_log_var_val['best_loss_test']=tf_log_var_val['loss_test']
                tf_log_var_val['best_loss_test_epoch']=epoch+1
            if tf_log_var_val['top1_accuracy_test']>tf_log_var_val['best_top1_accuracy_test']:
                tf_log_var_val['best_top1_accuracy_test']=tf_log_var_val['top1_accuracy_test']
                tf_log_var_val['best_top1_accuracy_test_epoch']=epoch+1
            if tf_log_var_val['top5_accuracy_test']>tf_log_var_val['best_top5_accuracy_test']:
                tf_log_var_val['best_top5_accuracy_test']=tf_log_var_val['top5_accuracy_test']
                tf_log_var_val['best_top5_accuracy_test_epoch']=epoch+1
        
            tf_log_var_names_assign_ops=[tf_variables[v+'_var'].assign(tf_log_var_val[v]) for v in tf_log_var_names]
            session.run(tf_log_var_names_assign_ops)
            for v in tf_log_var_names:
                value=tf_log_var_val[v]
                if isinstance(value,(np.int32,np.int64)):
                    history[iteration][v].append(int(value))
                elif isinstance(value,(np.float32,np.float64)):
                    history[iteration][v].append(float(value))
                else:
                    history[iteration][v].append(value)
            if save_history:
                with open(history_file,'w') as f:
                    json.dump(history,f,indent=4)
            if save_session and epoch%save_session_freq==0:
                saver.save(session,os.path.join(checkpoint_dir,'my_model'))
                if verbose>=2:
                    print('written epoch %d, global step %d'%(epoch+1,tf_log_var_val['global_step']))

            if evaluation_metric=='top1_accuracy':
                new_metric_val=tf_log_var_val['top1_accuracy_test']
            elif evaluation_metric=='top5_accuracy':
                new_metric_val=tf_log_var_val['top5_accuracy_test']
            elif evaluation_metric=='validation_loss':
                new_metric_val=-tf_log_var_val['loss_test']

            if new_metric_val>best_metric_val:
                print('saving model parameters...')
                tf_networks['train_network'].save_model(best_model_params_file,session)
                for k in tf_log_var_val:
    #                 print(k+' '+str(tf_log_var_val[k])+' '+str(type(tf_log_var_val[k])))
                    if isinstance(tf_log_var_val[k],np.int64):
                        tf_log_var_val[k]=int(tf_log_var_val[k])
                    if isinstance(tf_log_var_val[k],np.int32):
                        tf_log_var_val[k]=int(tf_log_var_val[k])
                    if isinstance(tf_log_var_val[k],np.float32):
                        tf_log_var_val[k]=float(tf_log_var_val[k])

                with open(best_model_eval_metric_file,'w') as f:
                    json.dump(tf_log_var_val,f,indent=4)
        if verbose>=1:
            print('On iteration %d'%(iteration+1))
            print('\tFinal best top1 validation accuracy: %f'%tf_log_var_val['best_top1_accuracy_test'])
            print('\tFinal best top5 validation accuracy: %f'%tf_log_var_val['best_top5_accuracy_test'])
            print('\tFinal best validation loss: %f'%tf_log_var_val['best_loss_test'])
    #Final evaluation
    print('===========Final Evaluation=============')
    #evaluation on trainset
    evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                            class_loss=tf_tensors['class_loss_test'],
                            regularization_loss=tf_tensors['top1_accuracy_test'],
                            top1_accuracy=tf_tensors['top1_accuracy_test'],
                            top5_accuracy=tf_tensors['top5_accuracy_test'],
                            X=tf_tensors['X_test'],
                            Y=tf_tensors['Y_test'])
    evaluation_vals=test_accuracy_evaluation_plain(data_dict_total['X_train'],
                                                   data_dict_total['Y_train'],
                                                   hyper_params['test_batch_size'],
                                                   evaluation_tensors,session)
    tf_log_var_val.update({k+'_train':evaluation_vals[k] for k in evaluation_vals})
    #evaluation on testset
    evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                            class_loss=tf_tensors['class_loss_test'],
                            regularization_loss=tf_tensors['regularization_loss_test'],
                            top1_accuracy=tf_tensors['top1_accuracy_test'],
                            top5_accuracy=tf_tensors['top5_accuracy_test'],
                            X=tf_tensors['X_test'],
                            Y=tf_tensors['Y_test'])
    evaluation_vals=test_accuracy_evaluation_plain(data_dict_total['X_test'],
                                                   data_dict_total['Y_test'],
                                                   hyper_params['test_batch_size'],
                                                   evaluation_tensors,session)
    tf_log_var_val.update({k+'_test':evaluation_vals[k] for k in evaluation_vals})        


    if verbose>=2:
        print_var_list=['loss_train','class_loss_train','regularization_loss_train',
                       'loss_test','class_loss_test','regularization_loss_test',
                       'top1_accuracy_train','top5_accuracy_train','top1_accuracy_test','top5_accuracy_test']
        print_eval_metric(tf_log_var_val,print_var_list,'Final evaluation on all classes') 
    train_writer.close()
    test_writer.close()