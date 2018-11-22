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
from icarl_utils import *
from scipy.spatial.distance import cdist
import time
np.random.seed(1997)
def build_graph_myil(hyper_params,fixed_params):
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
    tf_tensors['sample_weight_train']=tf.placeholder(tf.float32,shape=[None])
    tf_tensors['Y_one_hot_train']=tf.one_hot(tf_tensors['Y_train'],fixed_params['total_num_classes'])
    tf_tensors['class_loss_train']=tf.reduce_sum(
               tf.nn.sigmoid_cross_entropy_with_logits(
               labels=tf_tensors['Y_one_hot_train'],
                logits=tf_networks['train_network'].tf_tensors['fc'])*\
                tf.reshape(tf_tensors['sample_weight_train'],[-1,1]))/\
                (tf.reduce_sum(tf_tensors['sample_weight_train'])*fixed_params['total_num_classes'])
    tf_tensors['class_loss_train_old']=tf.reduce_mean(
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
    tf_tensors['batch_class_loss_test']=tf.reduce_mean(
               tf.nn.sigmoid_cross_entropy_with_logits(
               labels=tf_tensors['Y_one_hot_test'],logits=tf_networks['test_network'].tf_tensors['fc']),axis=1)
    tf_tensors['class_loss_test']=tf.reduce_mean(tf_tensors['batch_class_loss_test'],axis=0)
    tf_tensors['regularization_loss_test']=hyper_params['beta']*tf_networks['test_network'].tf_tensors['l2_loss']
    #print(tf_networks['test_network'].tf_tensors['l2_loss'])
    tf_tensors['loss_test']=tf_tensors['class_loss_test']+tf_tensors['regularization_loss_test']
    tf_tensors['feature_map_test']=tf_networks['test_network'].tf_tensors['pool_last'][:,0,0,:]
    indices_of_ranks_test=tf.nn.top_k(-tf_networks['test_network'].tf_tensors['fc'],k=fixed_params['total_num_classes'])[1]
    ranks_of_indices_test=tf.nn.top_k(-indices_of_ranks_test,k=fixed_params['total_num_classes'])[1]
    tf_tensors['ranks_of_indices_test']=ranks_of_indices_test
    indexing_matrix_test=tf.stack([tf.range(tf.shape(tf_tensors['Y_test'])[0]),tf_tensors['Y_test']],axis=1)
    ranks_of_groud_truth_class_test=fixed_params['total_num_classes']-1-tf.gather_nd(ranks_of_indices_test,indexing_matrix_test)
    tf_tensors['ranks_of_groud_truth_class_test']=ranks_of_groud_truth_class_test
    tf_tensors['top1_accuracy_test']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_test,1),tf.float32))
    tf_tensors['top5_accuracy_test']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_test,5),tf.float32))
    
    tf_networks['prev_network']=ResNet('ResNet32','test',num_outputs=fixed_params['total_num_classes'],name='ResNet32_previous')
    tf_tensors['prev_network_sigmoid']=tf.sigmoid(tf_networks['prev_network'].tf_tensors['fc'])
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
def fit_myil(tf_tensors,tf_variables,tf_networks,fixed_params,hyper_params,data_dict_total,session,resume=False,
        save_session=True,save_session_freq=1,save_params=True,evaluation_metric='top1_accuracy',save_history=True,
        num_epochs=40,num_iterations=10,verbose=2,print_freq=1,pretrain_evaluation=1,override_warning=True):
    tf_log_var_names=['local_step','global_step','iteration_step',
                'loss_train','loss_test','best_loss_test','best_loss_test_epoch',
                'top1_accuracy_train','top1_accuracy_test','best_top1_accuracy_test','best_top1_accuracy_test_epoch',
                'top5_accuracy_train','top5_accuracy_test','best_top5_accuracy_test','best_top5_accuracy_test_epoch']
    other_log_var_names=['best_top1_accuracy_exemplar_mean_test','best_top1_accuracy_theoretical_mean_test',
                         'best_top5_accuracy_exemplar_mean_test','best_top5_accuracy_theoretical_mean_test']
    if verbose>=2:
        pprint(fixed_params)
        pprint(hyper_params)
        
    #determine class batch numbers
    max_num_iterations=np.ceil(fixed_params['total_num_classes']/fixed_params['class_batch_size'])
    if num_iterations>max_num_iterations:
        num_iterations=int(max_num_iterations)
        print('num_iterations exceeds maximum allowed number, reset to '+str(max_num_iterations))
        
    # get output file names
    tensorboard_dir=os.path.join(fixed_params['base_dir'],'tensorboard_dir')
    checkpoint_dir=os.path.join(fixed_params['base_dir'],'checkpoint_dir')
    hyper_params_file=os.path.join(fixed_params['base_dir'],'hyper_params.json')
    fixed_params_file=os.path.join(fixed_params['base_dir'],'fixed_params.json')
    history_file=os.path.join(fixed_params['base_dir'],'history.json')
    class_ord_file=os.path.join(fixed_params['base_dir'],'class_ord.json')
    exemplar_file=os.path.join(fixed_params['base_dir'],'exemplar.pkl')
    exemplar_mean_file=os.path.join(fixed_params['base_dir'],'exemplar_mean.pkl')
    theoretical_mean_file=os.path.join(fixed_params['base_dir'],'theoretical_mean.pkl')
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
        with open(exemplar_file,'rb') as f_exemp,open(exemplar_mean_file,'rb') as f_mean:
            exemplar=pickle.load(f_exemp)
            exemplar_mean=pickle.load(f_mean)
        if fixed_params['use_theoretical_mean']:
            with open(theoretical_mean_file,'rb') as f_theo:
                theoretical_mean=pickle.load(f_theo)
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
            for i in range(num_iterations):
                history[i].update(zip(other_log_var_names,[np.nan]*len(other_log_var_names)))
        #init class_ord
        class_ord=np.unique(np.append(data_dict_total['Y_train'],data_dict_total['Y_test']))
        assert(len(class_ord)>=num_iterations*fixed_params['class_batch_size'])
        np.random.shuffle(class_ord)
        class_ord=[int(i) for i in class_ord]
        with open(class_ord_file,'w') as f_class_ord:
            json.dump(class_ord,f_class_ord,indent=4)
        #init exemplar and class mean
        _,W,H,C=data_dict_total['X_train'].shape
        exemplar=np.ones([fixed_params['total_num_classes'],hyper_params['exemplar_set_size'],W,H,C])*np.nan
        feat_size=tf_tensors['feature_map_test'].shape.as_list()[1]
        exemplar_mean=np.ones([fixed_params['total_num_classes'],feat_size])*np.nan
        with open(exemplar_file,'wb') as f_exemp,open(exemplar_mean_file,'wb') as f_mean:
            pickle.dump(exemplar,f_exemp)
            pickle.dump(exemplar_mean,f_mean)
        if fixed_params['use_theoretical_mean']:
            theoretical_mean=np.ones([fixed_params['total_num_classes'],feat_size])*np.nan
            with open(theoretical_mean_file,'wb') as f_theo:
                pickle.dump(theoretical_mean,f_theo)
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
    # previous network
    previous_network=tf_networks['prev_network']
    for iteration in range(tf_log_var_val['iteration_step'],num_iterations):
        #train and test file for each class_batch
        using_classes=class_ord[iteration*fixed_params['class_batch_size']:(iteration+1)*fixed_params['class_batch_size']]
        train_idx=np.array([i in using_classes for i in data_dict_total['Y_train']])
        test_idx=np.array([i in using_classes for i in data_dict_total['Y_test']])
        data_dict=dict(X_train=data_dict_total['X_train'][train_idx],
                       Y_train=data_dict_total['Y_train'][train_idx],
                       sample_weight_train=np.ones_like(np.nonzero(train_idx)[0]),
                       X_test=data_dict_total['X_test'][test_idx],
                       Y_test=data_dict_total['Y_test'][test_idx])
        #cumul dataset
        using_classes=class_ord[:(iteration+1)*fixed_params['class_batch_size']]
        train_idx=np.array([i in using_classes for i in data_dict_total['Y_train']])
        test_idx=np.array([i in using_classes for i in data_dict_total['Y_test']])
        data_dict_cumul=dict(X_train=data_dict_total['X_train'][train_idx],
                               Y_train=data_dict_total['Y_train'][train_idx],
                               X_test=data_dict_total['X_test'][test_idx],
                               Y_test=data_dict_total['Y_test'][test_idx])  
        #ori dataset
        using_classes=class_ord[:fixed_params['class_batch_size']]
        train_idx=np.array([i in using_classes for i in data_dict_total['Y_train']])
        test_idx=np.array([i in using_classes for i in data_dict_total['Y_test']])
        data_dict_ori=dict(X_train=data_dict_total['X_train'][train_idx],
                               Y_train=data_dict_total['Y_train'][train_idx],
                               X_test=data_dict_total['X_test'][test_idx],
                               Y_test=data_dict_total['Y_test'][test_idx]) 
        using_classes=class_ord[iteration*fixed_params['class_batch_size']:(iteration+1)*fixed_params['class_batch_size']]
        #add exemplars
        if iteration>=1:
            previous_best_model_params_file=os.path.join(fixed_params['base_dir'],'best_model_params_%d.pkl'%(iteration-1))
            previous_network.load_model(previous_best_model_params_file,session)
            tf_networks['train_network'].load_model(previous_best_model_params_file,session)
            _,W,H,C=data_dict['X_train'].shape
            exemplars_up_to_now=exemplar[class_ord[:iteration*fixed_params['class_batch_size']],...].reshape([-1,W,H,C])
            assert(np.all(~np.isnan(exemplars_up_to_now)))
            exemplars_up_to_now_label=np.ones(len(exemplars_up_to_now),dtype=np.int32)
            pointer=0
            for c in class_ord[:iteration*fixed_params['class_batch_size']]:
                for i in range(pointer,pointer+hyper_params['exemplar_set_size']):
                    exemplars_up_to_now_label[i]=c
                pointer+=hyper_params['exemplar_set_size']
            assert(np.all(~np.isnan(exemplars_up_to_now_label)))
            data_dict['X_train']=np.concatenate([data_dict['X_train'],exemplars_up_to_now],axis=0)
            data_dict['sample_weight_train']=np.concatenate([np.ones_like(data_dict['Y_train']),
                                                             np.ones_like(exemplars_up_to_now_label)],axis=0)
            data_dict['Y_train']=np.concatenate([data_dict['Y_train'],exemplars_up_to_now_label],axis=0)
            
            
                                              
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
                #plain method evaluation
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
                    print_eval_metric(tf_log_var_val,print_var_list,'Class batch pretrain evaluation (plain method)')
#             #pre-collect training set
#             iterator=iterate_minibatches(data_dict['X_train'],data_dict['Y_train'],len(data_dict['X_train']),shuffle=True,augment=True)
#             X_collect,Y_collect=next(iterator)
#             assert next(iterator,False)==False
#             Y_one_hot_collect=np.zeros([len(Y_collect),fixed_params['total_num_classes']],dtype=np.float32)
#             Y_one_hot_collect[range(len(Y_one_hot_collect)),Y_collect]=1.
#             #distillation
#             if iteration>=1:
#                 prediction_old=session.run([tf.sigmoid(previous_network.tf_tensors['fc'])],
#                                            feed_dict={previous_network.tf_tensors['input']:X_collect})[0]
#                 Y_one_hot_collect[:,class_ord[:iteration*fixed_params['class_batch_size']]]=\
#                 prediction_old[:,class_ord[:iteration*fixed_params['class_batch_size']]]
#             #training
#             tf_log_var_val['loss_train']=0
#             tf_log_var_val['class_loss_train']=0
#             tf_log_var_val['regularization_loss_train']=0
#             tf_log_var_val['top1_accuracy_train']=0
#             tf_log_var_val['top5_accuracy_train']=0
#             num_batches_train=int((len(data_dict['X_train'])/hyper_params['train_batch_size'])) 
#             Y_ind=np.arange(Y_one_hot_collect.shape[0])
#             for X_minibatch,Y_ind_minibatch in iterate_minibatches(X_collect,Y_ind,hyper_params['train_batch_size'],shuffle=False,augment=False): #important to set shuffle and augment to False
#                 Y_one_hot_minibatch=Y_one_hot_collect[Y_ind_minibatch,...]
#                 Y_minibatch=data_dict['Y_train'][Y_ind_minibatch]
#                 _,local_loss,local_class_loss,local_regularization_loss=\
#                 session.run([tf_tensors['optimizer_train'],
#                              tf_tensors['loss_train'],
#                              tf_tensors['class_loss_train'],
#                              tf_tensors['regularization_loss_train']],
#                           feed_dict={tf_tensors['X_train']:X_minibatch,
#                                      tf_tensors['Y_one_hot_train']:Y_one_hot_minibatch})
#                 local_top1_accuracy,local_top5_accuracy=\
#                 session.run([tf_tensors['top1_accuracy_train'],
#                              tf_tensors['top5_accuracy_train']],
#                             feed_dict={tf_tensors['X_train']:X_minibatch,
#                                        tf_tensors['Y_train']:Y_minibatch})
#                 tf_log_var_val['local_step']+=1
#                 tf_log_var_val['loss_train']+=local_loss/num_batches_train
#                 tf_log_var_val['class_loss_train']+=local_class_loss/num_batches_train
#                 tf_log_var_val['regularization_loss_train']+=local_regularization_loss/num_batches_train
#                 tf_log_var_val['top1_accuracy_train']+=local_top1_accuracy/num_batches_train
#                 tf_log_var_val['top5_accuracy_train']+=local_top5_accuracy/num_batches_train

            #training (slow implementation)
            t=time.time()
            train_tensors=dict((k,tf_tensors[k]) for k in ['optimizer_train','loss_train','class_loss_train',
                                                           'regularization_loss_train','X_train','Y_one_hot_train',
                                                          'Y_train','top1_accuracy_train','sample_weight_train',
                                                           'top5_accuracy_train','prev_network_sigmoid'])
            
            #without distillation
        
            assert np.all(~np.isnan(data_dict['X_train'])) 
            eval_vals_train=train_with_sample_weight(data_dict['X_train'],data_dict['Y_train'],
                                                     data_dict['sample_weight_train'],
                                                     hyper_params['train_batch_size'],
                                                     train_tensors,
                                                     session)
#             eval_vals_train=train_plain(data_dict['X_train'],data_dict['Y_train'],
#                                                      hyper_params['train_batch_size'],
#                                                      train_tensors,
#                                                      session)
                   
            tf_log_var_val.update(dict((k,eval_vals_train[k]) for k in ['loss_train','class_loss_train','regularization_loss_train','top1_accuracy_train','top5_accuracy_train']))
            tf_log_var_val['local_step']+=eval_vals_train['local_step']
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
                    print('written epoch %d, global step %d, time %f'%(epoch+1,tf_log_var_val['global_step'],time.time()-t))
            else:
                if verbose>=2:
                    print('epoch %d, global step %d, time %f'%(epoch+1,tf_log_var_val['global_step'],time.time()-t))                

            if evaluation_metric=='top1_accuracy':
                new_metric_val=tf_log_var_val['top1_accuracy_test']
            elif evaluation_metric=='top5_accuracy':
                new_metric_val=tf_log_var_val['top5_accuracy_test']
            elif evaluation_metric=='validation_loss':
                new_metric_val=-tf_log_var_val['loss_test']

            if new_metric_val>best_metric_val or epoch==0:
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

        #plain evaluation
        final_best_model_params_file=os.path.join(fixed_params['base_dir'],'best_model_params_%d.pkl'%(iteration))
        tf_networks['test_network'].load_model(final_best_model_params_file,session)        
        plain_evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                class_loss=tf_tensors['class_loss_test'],
                                regularization_loss=tf_tensors['regularization_loss_test'],
                                top1_accuracy=tf_tensors['top1_accuracy_test'],
                                top5_accuracy=tf_tensors['top5_accuracy_test'],
                                X=tf_tensors['X_test'],
                                Y=tf_tensors['Y_test'])
        
        eval_vals_test=test_accuracy_evaluation_plain(data_dict['X_test'],
                                                       data_dict['Y_test'],
                                                       hyper_params['test_batch_size'],
                                                       plain_evaluation_tensors,session)  
        eval_vals_cumul=test_accuracy_evaluation_plain(data_dict_cumul['X_test'],
                                                       data_dict_cumul['Y_test'],
                                                       hyper_params['test_batch_size'],
                                                       plain_evaluation_tensors,session) 
        eval_vals_ori=test_accuracy_evaluation_plain(data_dict_ori['X_test'],
                                                       data_dict_ori['Y_test'],
                                                       hyper_params['test_batch_size'],
                                                       plain_evaluation_tensors,session) 
        
        if save_history:
            history[iteration].update(dict(best_top1_accuracy_plain_test=eval_vals_test['top1_accuracy'],
                                           best_top5_accuracy_plain_test=eval_vals_test['top5_accuracy'],
                                           best_top1_accuracy_plain_cumul=eval_vals_cumul['top1_accuracy'],
                                           best_top5_accuracy_plain_cumul=eval_vals_cumul['top5_accuracy'],
                                           best_top1_accuracy_plain_ori=eval_vals_ori['top1_accuracy'],
                                           best_top5_accuracy_plain_ori=eval_vals_ori['top5_accuracy']))
            
            with open(history_file,'w') as f:
                json.dump(history,f,indent=4)
                
        if verbose>=1:
            print('On iteration %d'%(iteration+1))
            print('Plain evaluation')
            print('\tBest top1 validation accuracy: %f'%eval_vals_test['top1_accuracy'])
            print('\tBest top5 validation accuracy: %f'%eval_vals_test['top5_accuracy'])
            print('\tBest top1 cumul accuracy: %f'%eval_vals_cumul['top1_accuracy'])
            print('\tBest top5 cumul accuracy: %f'%eval_vals_cumul['top5_accuracy'])
            print('\tBest top1 ori accuracy: %f'%eval_vals_ori['top1_accuracy'])
            print('\tBest top5 ori accuracy: %f'%eval_vals_ori['top5_accuracy'])  
            
        
        #update exemplars
        feature_map_tensors=dict(X=tf_tensors['X_test'],feature_map=tf_tensors['feature_map_test'])
        exemplar_set_update_tensors=dict(X=tf_tensors['X_test'],
                                         Y=tf_tensors['Y_test'],
                                         batch_loss=tf_tensors['batch_class_loss_test'])         
        update_exemplar_set_by_loss(exemplar,data_dict,class_ord[iteration*fixed_params['class_batch_size']:(iteration+1)*fixed_params['class_batch_size']],hyper_params['test_batch_size'],hyper_params['exemplar_set_size'],exemplar_set_update_tensors,session,fixed_params['base_dir'])
        with open(exemplar_file,'wb') as f:
            pickle.dump(exemplar,f)
        #update exemplar_mean
        current_exemplars=exemplar[class_ord[:((iteration+1)*fixed_params['class_batch_size'])]]
        CL,PR,W,H,C=current_exemplars.shape
        assert(np.all(~np.isnan(current_exemplars)))
        feature_maps=test_get_feature_maps(current_exemplars.reshape([-1,W,H,C]),
                                                    hyper_params['test_batch_size'],
                                           feature_map_tensors,
                                           session,shuffle=False,yield_remaining=True)\
        .reshape([CL,PR,-1])
        unnormalized_exemplar_mean=feature_maps.mean(axis=1)
        exemplar_mean[class_ord[:(iteration+1)*fixed_params['class_batch_size']]]=\
        unnormalized_exemplar_mean/np.linalg.norm(unnormalized_exemplar_mean,axis=1)[...,np.newaxis]
        with open(exemplar_mean_file,'wb') as f:
            pickle.dump(exemplar_mean,f)
        #exemplar mean evaluation
        assert(np.all(~np.isnan(exemplar_mean[class_ord[:(iteration+1)*fixed_params['class_batch_size']],...])))
        eval_vals_test=test_accuracy_evaluation_ncm(data_dict['X_test'],data_dict['Y_test'],
                                               exemplar_mean,hyper_params['test_batch_size'],feature_map_tensors,session)
        eval_vals_cumul=test_accuracy_evaluation_ncm(data_dict_cumul['X_test'],data_dict_cumul['Y_test'],
                                               exemplar_mean,hyper_params['test_batch_size'],feature_map_tensors,session)
        eval_vals_ori=test_accuracy_evaluation_ncm(data_dict_ori['X_test'],data_dict_ori['Y_test'],
                                               exemplar_mean,hyper_params['test_batch_size'],feature_map_tensors,session)        
        if save_history:
            history[iteration].update(dict(best_top1_accuracy_exemplar_mean_test=eval_vals_test['top1_accuracy'],
                                           best_top5_accuracy_exemplar_mean_test=eval_vals_test['top5_accuracy'],
                                           best_top1_accuracy_exemplar_mean_cumul=eval_vals_cumul['top1_accuracy'],
                                           best_top5_accuracy_exemplar_mean_cumul=eval_vals_cumul['top5_accuracy'],
                                           best_top1_accuracy_exemplar_mean_ori=eval_vals_ori['top1_accuracy'],
                                           best_top5_accuracy_exemplar_mean_ori=eval_vals_ori['top5_accuracy']))
            with open(history_file,'w') as f:
                json.dump(history,f,indent=4)
        if verbose>=1:
            print('Exemplar mean evaluation')
            print('\tBest top1 validation accuracy: %f'%eval_vals_test['top1_accuracy'])
            print('\tBest top5 validation accuracy: %f'%eval_vals_test['top5_accuracy'])
            print('\tBest top1 cumul accuracy: %f'%eval_vals_cumul['top1_accuracy'])
            print('\tBest top5 cumul accuracy: %f'%eval_vals_cumul['top5_accuracy'])
            print('\tBest top1 ori accuracy: %f'%eval_vals_ori['top1_accuracy'])
            print('\tBest top5 ori accuracy: %f'%eval_vals_ori['top5_accuracy'])            
        #update theoretical_mean
        if fixed_params['use_theoretical_mean']:
            
            prev_classes=class_ord[:((iteration+1)*fixed_params['class_batch_size'])]
            prev_ind=[i in prev_classes for i in data_dict_total['Y_train']]
            X_prev=data_dict_total['X_train'][prev_ind]
            Y_prev=data_dict_total['Y_train'][prev_ind]
            feature_maps=test_get_feature_maps(X_prev,hyper_params['test_batch_size'],feature_map_tensors,session,shuffle=False,yield_remaining=True)
            for c in prev_classes:
                unnormalized_theoretical_mean=feature_maps[Y_prev==c].mean(axis=0)
                theoretical_mean[c,:]=unnormalized_theoretical_mean/np.linalg.norm(unnormalized_theoretical_mean)
            with open(theoretical_mean_file,'wb') as f:
                pickle.dump(theoretical_mean,f)
            #theoretical mean evaluation
            assert(np.all(~np.isnan(theoretical_mean[class_ord[:(iteration+1)*fixed_params['class_batch_size']],...])))
            eval_vals_test=test_accuracy_evaluation_ncm(data_dict['X_test'],data_dict['Y_test'],
                                                   theoretical_mean,hyper_params['test_batch_size'],feature_map_tensors,session)
            eval_vals_cumul=test_accuracy_evaluation_ncm(data_dict_cumul['X_test'],data_dict_cumul['Y_test'],
                                                   theoretical_mean,hyper_params['test_batch_size'],feature_map_tensors,session)
            eval_vals_ori=test_accuracy_evaluation_ncm(data_dict_ori['X_test'],data_dict_ori['Y_test'],
                                               theoretical_mean,hyper_params['test_batch_size'],feature_map_tensors,session) 
            if save_history:
                history[iteration].update(dict(best_top1_accuracy_theoretical_mean_test=eval_vals_test['top1_accuracy'],
                                               best_top5_accuracy_theoretical_mean_test=eval_vals_test['top5_accuracy'],
                                               best_top1_accuracy_theoretical_mean_cumul=eval_vals_cumul['top1_accuracy'],
                                               best_top5_accuracy_theoretical_mean_cumul=eval_vals_cumul['top5_accuracy'],
                                               best_top1_accuracy_theoretical_mean_ori=eval_vals_ori['top1_accuracy'],
                                               best_top5_accuracy_theoretical_mean_ori=eval_vals_ori['top5_accuracy']))
                with open(history_file,'w') as f:
                    json.dump(history,f,indent=4)            
            if verbose>=1:
                print('Theoretical mean evaluation')
                print('\tBest top1 validation accuracy: %f'%eval_vals_test['top1_accuracy'])
                print('\tBest top5 validation accuracy: %f'%eval_vals_test['top5_accuracy'])
                print('\tBest top1 cumul accuracy: %f'%eval_vals_cumul['top1_accuracy'])
                print('\tBest top5 cumul accuracy: %f'%eval_vals_cumul['top5_accuracy'])
                print('\tBest top1 ori accuracy: %f'%eval_vals_ori['top1_accuracy'])
                print('\tBest top5 ori accuracy: %f'%eval_vals_ori['top5_accuracy'])      
    #Final retrain with exemplars

    #Final evaluation

    print('===========Final Evaluation=============')
    final_best_model_params_file=os.path.join(fixed_params['base_dir'],'best_model_params_%d.pkl'%(num_iterations-1))
    tf_networks['test_network'].load_model(final_best_model_params_file,session)
                  
    #plain evaluation
    #evaluation on trainset
    plain_evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                            class_loss=tf_tensors['class_loss_test'],
                            regularization_loss=tf_tensors['top1_accuracy_test'],
                            top1_accuracy=tf_tensors['top1_accuracy_test'],
                            top5_accuracy=tf_tensors['top5_accuracy_test'],
                            X=tf_tensors['X_test'],
                            Y=tf_tensors['Y_test'])
    evaluation_vals=test_accuracy_evaluation_plain(data_dict_total['X_train'],
                                                   data_dict_total['Y_train'],
                                                   hyper_params['test_batch_size'],
                                                   plain_evaluation_tensors,session)
                  
    tf_log_var_val.update({k+'_train':evaluation_vals[k] for k in evaluation_vals})
                  
                  
    #evaluation on testset
    plain_evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                            class_loss=tf_tensors['class_loss_test'],
                            regularization_loss=tf_tensors['regularization_loss_test'],
                            top1_accuracy=tf_tensors['top1_accuracy_test'],
                            top5_accuracy=tf_tensors['top5_accuracy_test'],
                            X=tf_tensors['X_test'],
                            Y=tf_tensors['Y_test'])
    evaluation_vals=test_accuracy_evaluation_plain(data_dict_total['X_test'],
                                                   data_dict_total['Y_test'],
                                                   hyper_params['test_batch_size'],
                                                   plain_evaluation_tensors,session)
    tf_log_var_val.update({k+'_test':evaluation_vals[k] for k in evaluation_vals})        
    if save_history:
        history.append(dict(final_top1_accuracy_plain_train=tf_log_var_val['top1_accuracy_train'],
                            final_top1_accuracy_plain_test=tf_log_var_val['top1_accuracy_test'],
                            final_top5_accuracy_plain_train=tf_log_var_val['top5_accuracy_train'],
                            final_top5_accuracy_plain_test=tf_log_var_val['top5_accuracy_test']
                            ))
        with open(history_file,'w') as f:
            json.dump(history,f,indent=4)
    if verbose>=2:
        print_var_list=['loss_train','class_loss_train','regularization_loss_train',
                       'loss_test','class_loss_test','regularization_loss_test',
                       'top1_accuracy_train','top5_accuracy_train','top1_accuracy_test','top5_accuracy_test']
        print_eval_metric(tf_log_var_val,print_var_list,'Final evaluation on all classes (plain method)') 
    
    #exemplar mean evaluation
    assert(np.all(~np.isnan(exemplar_mean[class_ord[:(iteration+1)*fixed_params['class_batch_size']],...])))
    feature_map_tensors=dict(X=tf_tensors['X_test'],feature_map=tf_tensors['feature_map_test'])
    eval_vals=test_accuracy_evaluation_ncm(data_dict_total['X_test'],data_dict_total['Y_test'],
                                           exemplar_mean,hyper_params['test_batch_size'],feature_map_tensors,session)
    if save_history:
        history[-1].update(dict(final_top1_accuracy_exemplar_mean_test=eval_vals['top1_accuracy'],
                                       final_top5_accuracy_exemplar_mean_test=eval_vals['top5_accuracy']))
        with open(history_file,'w') as f:
            json.dump(history,f,indent=4)
    if verbose>=2:
        print('Final evaluation on all classes (Exemplar mean)')
        print('\tBest top1 validation accuracy: %f'%eval_vals['top1_accuracy'])
        print('\tBest top5 validation accuracy: %f'%eval_vals['top5_accuracy'])
    
    #theoretical mean evaluation
    if fixed_params['use_theoretical_mean']:
        assert(np.all(~np.isnan(theoretical_mean[class_ord[:(iteration+1)*fixed_params['class_batch_size']],...])))
        feature_map_tensors=dict(X=tf_tensors['X_test'],feature_map=tf_tensors['feature_map_test'])
        eval_vals=test_accuracy_evaluation_ncm(data_dict_total['X_test'],data_dict_total['Y_test'],
                                               theoretical_mean,hyper_params['test_batch_size'],feature_map_tensors,session)
        if save_history:
            history[-1].update(dict(final_top1_accuracy_theoretical_mean_test=eval_vals['top1_accuracy'],
                                           final_top5_accuracy_theoretical_mean_test=eval_vals['top5_accuracy']))
            with open(history_file,'w') as f:
                json.dump(history,f,indent=4)
        if verbose>=2:
            print('Final evaluation on all classes (Theoretical mean)')
            print('\tBest top1 validation accuracy: %f'%eval_vals['top1_accuracy'])
            print('\tBest top5 validation accuracy: %f'%eval_vals['top5_accuracy'])    
        train_writer.close()
        test_writer.close()
