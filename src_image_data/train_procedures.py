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
import matplotlib.pyplot as plt
import json
from pprint import pprint
from scipy.spatial.distance import cdist
import time
from copy import copy
import warnings
import sys
from train_utils import *
from nn_lib import *
from cifar100_utils import *
np.random.seed(1997)
def build_graph(hyper_params,fixed_params):
    tf.reset_default_graph()
    if 'random_seed' in fixed_params:
        tf.set_random_seed(fixed_params['random_seed'])
    tf_tensors={}
    tf_variables={}
    tf_networks={}
    if fixed_params['dataset'] in ['cifar100','breakhis','cifar10']:
        tf_networks['network_train']=\
        ResNet(fixed_params['net_type'],'train',num_outputs=fixed_params['total_num_classes'],name='network_train',se=hyper_params['se'],
              use_fisher=True)
    elif fixed_params['dataset'] in ['hela10','mnist']:
        tf_networks['network_train']=\
        ResNet(fixed_params['net_type'],'train',num_outputs=fixed_params['total_num_classes'],name='network_train',se=hyper_params['se'],
              input_channels=1,use_fisher=True)
    else:
        assert False
    tf_tensors['X_train']=tf_networks['network_train'].tf_tensors['input']
    tf_tensors['Y_train']=tf.placeholder(tf.int32,shape=[None])

    tf_tensors['Y_one_hot_train']=tf.one_hot(tf_tensors['Y_train'],fixed_params['total_num_classes'])

    assert hyper_params['loss_function'] in ['sigmoid_cross_entropy_with_logits','softmax_cross_entropy_with_logits']
    if hyper_params['loss_function']=='sigmoid_cross_entropy_with_logits' and hyper_params['ewc_lambda']>0:
        warnings.warn('when use ewc, loss_function should be softmax')
    elif hyper_params['loss_function']=='softmax_cross_entropy_with_logits' and hyper_params['train_method']=='train_distillation':
        warnings.warn('when use distillation, loss_function should be sigmoid')
    if hyper_params['train_method']=='train_with_sample_weight':
        tf_tensors['sample_weight_train']=tf.placeholder(tf.float32,shape=[None])
        if hyper_params['loss_function']=='sigmoid_cross_entropy_with_logits':
            tf_tensors['class_loss_train']=tf.reduce_sum(
                       tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_tensors['Y_one_hot_train'],logits=tf_networks['network_train'].tf_tensors['fc'])*\
                        tf.reshape(tf_tensors['sample_weight_train'],[-1,1]))/\
                        (tf.reduce_sum(tf_tensors['sample_weight_train'])*fixed_params['total_num_classes'])
        elif hyper_params['loss_function']=='softmax_cross_entropy_with_logits':

            tf_tensors['class_loss_train']=tf.reduce_sum(
                       tf.nn.softmax_cross_entropy_with_logits(
                       labels=tf_tensors['Y_one_hot_train'],
                        logits=tf_networks['network_train'].tf_tensors['fc'])*\
                        tf_tensors['sample_weight_train'])/\
                        (tf.reduce_sum(tf_tensors['sample_weight_train']))
    else:
        if hyper_params['loss_function']=='sigmoid_cross_entropy_with_logits':
            tf_tensors['class_loss_train']=tf.reduce_mean(
                   tf.nn.sigmoid_cross_entropy_with_logits(
                   labels=tf_tensors['Y_one_hot_train'],logits=tf_networks['network_train'].tf_tensors['fc']))
        elif hyper_params['loss_function']=='softmax_cross_entropy_with_logits':
            tf_tensors['class_loss_train']=tf.reduce_mean(
                       tf.nn.softmax_cross_entropy_with_logits(
                       labels=tf_tensors['Y_one_hot_train'],logits=tf_networks['network_train'].tf_tensors['fc']))


    tf_tensors['regularization_loss_train']=hyper_params['beta']*tf_networks['network_train'].tf_tensors['l2_loss']
    tf_tensors['ewc_loss_train']=hyper_params['ewc_lambda']*tf_networks['network_train'].tf_tensors['ewc_loss']
    tf_tensors['loss_train']=tf_tensors['class_loss_train']+tf_tensors['regularization_loss_train']+tf_tensors['ewc_loss_train']
    tf_variables['lr']=tf.get_variable('lr',initializer=tf.constant(hyper_params['initial_lr'],dtype=tf.float32),
                                       trainable=False,dtype=tf.float32)
    if hyper_params['optimizer']=='momentum':
        tf_tensors['optimizer_train']=tf.train.MomentumOptimizer(learning_rate=tf_variables['lr'],momentum=0.9).\
        minimize(tf_tensors['loss_train'])
    elif hyper_params['optimizer']=='adam':
        tf_tensors['optimizer_train']=tf.train.AdamOptimizer().\
        minimize(tf_tensors['loss_train'])
    else:
        assert False
    tf_tensors['optimizer_fc_train']=\
        tf.train.AdamOptimizer().\
        minimize(tf_tensors['loss_train'],var_list=[tf_networks['network_train'].tf_variables['fc/W'],
                                                    tf_networks['network_train'].tf_variables['fc/b']])

    indices_of_ranks_train=tf.nn.top_k(-tf_networks['network_train'].tf_tensors['fc'],k=fixed_params['total_num_classes'])[1]
    ranks_of_indices_train=tf.nn.top_k(-indices_of_ranks_train,k=fixed_params['total_num_classes'])[1]
    tf_tensors['ranks_of_indices_train']=ranks_of_indices_train
    indexing_matrix_train=tf.stack([tf.range(tf.shape(tf_tensors['Y_train'])[0]),tf_tensors['Y_train']],axis=1)
    ranks_of_groud_truth_class_train=fixed_params['total_num_classes']-1-tf.gather_nd(ranks_of_indices_train,indexing_matrix_train)
    tf_tensors['ranks_of_groud_truth_class_train']=ranks_of_groud_truth_class_train
    tf_tensors['top1_accuracy_train']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_train,1),tf.float32))
    tf_tensors['top5_accuracy_train']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_train,5),tf.float32))
    #lizx: for_debug
    if fixed_params['dataset'] in ['cifar100','breakhis','cifar10']:
        tf_networks['network_test']=\
        ResNet(fixed_params['net_type'],'test',num_outputs=fixed_params['total_num_classes'],
               name='network_test',se=hyper_params['se'],use_fisher=True)
    elif fixed_params['dataset'] in ['hela10','mnist']:
        tf_networks['network_test']=ResNet(fixed_params['net_type'],'test',
                                       num_outputs=fixed_params['total_num_classes'],name='network_test',
                                       se=hyper_params['se'],input_channels=1,use_fisher=True)
    tf_tensors['X_test']=tf_networks['network_test'].tf_tensors['input']
    tf_tensors['Y_test']=tf.placeholder(tf.int32,shape=[None])
    tf_tensors['Y_one_hot_test']=tf.one_hot(tf_tensors['Y_test'],fixed_params['total_num_classes'])
    if hyper_params['loss_function']=='sigmoid_cross_entropy_with_logits':
        tf_tensors['class_loss_test']=tf.reduce_mean(
               tf.nn.sigmoid_cross_entropy_with_logits(
               labels=tf_tensors['Y_one_hot_test'],logits=tf_networks['network_test'].tf_tensors['fc']))/fixed_params['total_num_classes']
    elif hyper_params['loss_function']=='softmax_cross_entropy_with_logits':
        tf_tensors['class_loss_test']=tf.reduce_mean(
                   tf.nn.softmax_cross_entropy_with_logits(
                   labels=tf_tensors['Y_one_hot_test'],logits=tf_networks['network_test'].tf_tensors['fc']))

    tf_tensors['regularization_loss_test']=hyper_params['beta']*tf_networks['network_test'].tf_tensors['l2_loss']
    #print(tf_networks['network_test'].tf_tensors['l2_loss'])
    tf_tensors['ewc_loss_test']=hyper_params['ewc_lambda']*tf_networks['network_test'].tf_tensors['ewc_loss']
    tf_tensors['loss_test']=tf_tensors['class_loss_test']+tf_tensors['regularization_loss_test']+tf_tensors['ewc_loss_test']
    tf_tensors['feature_map_test']=tf_networks['network_test'].tf_tensors['pool_last'][:,0,0,:]
    indices_of_ranks_test=tf.nn.top_k(-tf_networks['network_test'].tf_tensors['fc'],k=fixed_params['total_num_classes'])[1]
    ranks_of_indices_test=tf.nn.top_k(-indices_of_ranks_test,k=fixed_params['total_num_classes'])[1]
    tf_tensors['ranks_of_indices_test']=ranks_of_indices_test
    indexing_matrix_test=tf.stack([tf.range(tf.shape(tf_tensors['Y_test'])[0]),tf_tensors['Y_test']],axis=1)
    ranks_of_groud_truth_class_test=fixed_params['total_num_classes']-1-tf.gather_nd(ranks_of_indices_test,indexing_matrix_test)
    tf_tensors['ranks_of_groud_truth_class_test']=ranks_of_groud_truth_class_test
    tf_tensors['top1_accuracy_test']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_test,1),tf.float32))
    tf_tensors['top5_accuracy_test']=tf.reduce_mean(tf.cast(tf.less(ranks_of_groud_truth_class_test,5),tf.float32))

    if fixed_params['dataset'] in ['cifar100','breakhis','cifar10']:
        tf_networks['network_prev']=\
        ResNet(fixed_params['net_type'],'test',num_outputs=fixed_params['total_num_classes'],name='network_prev',
               se=hyper_params['se'],use_fisher=True)
    elif fixed_params['dataset'] in ['hela10','mnist']:
        tf_networks['network_prev']=ResNet(fixed_params['net_type'],'test',
                                           num_outputs=fixed_params['total_num_classes'],name='network_prev',
                                           se=hyper_params['se'],input_channels=1,use_fisher=True)
    tf_tensors['network_prev_sigmoid']=tf.sigmoid(tf_networks['network_prev'].tf_tensors['fc'])

    #memory profile
    mbi_op=tf.contrib.memory_stats.MaxBytesInUse()
    tf_tensors['mbi_op']=mbi_op
    return tf_tensors,tf_variables,tf_networks

def fit(tf_tensors,tf_variables,tf_networks,fixed_params,hyper_params,data_dict_total,session,resume=False,
        save_params=True,evaluation_metric='top1_accuracy',save_history=True,
        num_epochs=40,num_iterations=10,verbose=2,print_freq=1,pretrain_evaluation=1,override_warning=True):

    # get variable names

    dataset=fixed_params['dataset']
    iteration_counter_var_names=['iteration_step']
    iteration_evaluation_var_names=[]

    if verbose>=2:
        pprint(fixed_params)
        pprint(hyper_params)

    # determine class batch numbers

    max_num_iterations=np.ceil(fixed_params['total_num_classes']/fixed_params['class_batch_size'])
    if num_iterations>max_num_iterations:
        num_iterations=int(max_num_iterations)
        print('num_iterations exceeds maximum allowed number, reset to '+str(max_num_iterations))

    # get output file names

    tensorboard_dir=os.path.join(fixed_params['base_dir'],'tensorboard_dir')
    hyper_params_file=os.path.join(fixed_params['base_dir'],'hyper_params.json')
    fixed_params_file=os.path.join(fixed_params['base_dir'],'fixed_params.json')
    history_file=os.path.join(fixed_params['base_dir'],'history.pkl')
    class_ord_file=os.path.join(fixed_params['base_dir'],'class_ord.json')
    icarl_exemplars_file=os.path.join(fixed_params['base_dir'],'icarl_exemplars.pkl')
    svm_exemplars_file=os.path.join(fixed_params['base_dir'],'svm_exemplars.pkl')
    icarl_exemplars_mean_file=os.path.join(fixed_params['base_dir'],'icarl_exemplars_mean.pkl')
    theoretical_mean_file=os.path.join(fixed_params['base_dir'],'theoretical_mean.pkl')

    # initialization 1: dirs

    if os.path.exists(fixed_params['base_dir']):
        if override_warning:
            choice=input(fixed_params['base_dir']+' already exists, override?')
        else:
            choice='n'
        if choice=='y':
            shutil.rmtree(fixed_params['base_dir'])
            os.makedirs(fixed_params['base_dir'])
        else:
            print('dir already exists, cancelled')
            return
    else:
            os.makedirs(fixed_params['base_dir'])

    # initialization 2: tensorflow global variables

    init=tf.global_variables_initializer()
    session.run(init)
    tf_networks['network_test'].set_model_params(tf_networks['network_train'].get_all_model_params(session),session)

    # dump hyper_params and fixed_params

    with open(hyper_params_file,'w') as f_hyper, open(fixed_params_file,'w') as f_fixed:
        json.dump(hyper_params,f_hyper,indent=4)
        json.dump(fixed_params,f_fixed,indent=4)

    # initialization 3: history

    history=[]

    # initialization 4: class_ord

    class_ord=np.unique(np.append(data_dict_total['Y_train'],data_dict_total['Y_test']))
    assert(len(class_ord)>=num_iterations*fixed_params['class_batch_size'])
    if hyper_params['shuffle_class_ord']:
        np.random.shuffle(class_ord)
    class_ord=[int(i) for i in class_ord]
    with open(class_ord_file,'w') as f_class_ord:
        json.dump(class_ord,f_class_ord,indent=4)

    # initialization 5: exemplar and class mean

    _,W,H,C=data_dict_total['X_train'].shape
    feat_size=tf_tensors['feature_map_test'].shape.as_list()[1]

    if not hyper_params['use_fixedsize_exemplars']:
        icarl_exemplars,svm_exemplars=initialize_exemplars(W,H,C,
                                                        fixed_params['total_num_classes'],hyper_params['exemplars_set_size'],dataset)

    icarl_exemplars_mean=initialize_icarl_exemplars_mean(fixed_params['total_num_classes'],feat_size)
    if fixed_params['use_theoretical_mean']:
        theoretical_mean=initialize_icarl_theoretical_mean(fixed_params['total_num_classes'],feat_size)

    # initialization 6: log_vars_preiteration

    log_vars_preiteration={}

    #pre-train evaluation on all class

    if  pretrain_evaluation>=2:
        #evaluation on trainset
        evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                class_loss=tf_tensors['class_loss_test'],
                                regularization_loss=tf_tensors['regularization_loss_test'],
                                top1_accuracy=tf_tensors['top1_accuracy_test'],
                                top5_accuracy=tf_tensors['top5_accuracy_test'],
                                X=tf_tensors['X_test'],
                                Y=tf_tensors['Y_test'],
                                fc=tf_networks['network_test'].tf_tensors['fc'])
        evaluation_vals=test_accuracy_evaluation_plain(data_dict_total['X_train'],
                                                       data_dict_total['Y_train'],
                                                       hyper_params['test_batch_size'],
                                                       evaluation_tensors,session,dataset=dataset)
        add_to_dict(log_vars_preiteration,evaluation_vals,'train',list(evaluation_vals.keys()))

        #evaluation on testset
        evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                class_loss=tf_tensors['class_loss_test'],
                                regularization_loss=tf_tensors['regularization_loss_test'],
                                top1_accuracy=tf_tensors['top1_accuracy_test'],
                                top5_accuracy=tf_tensors['top5_accuracy_test'],
                                X=tf_tensors['X_test'],
                                Y=tf_tensors['Y_test'],
                                fc=tf_networks['network_test'].tf_tensors['fc'])
        evaluation_vals=test_accuracy_evaluation_plain(data_dict_total['X_test'],
                                                       data_dict_total['Y_test'],
                                                       hyper_params['test_batch_size'],
                                                       evaluation_tensors,session,dataset=dataset)
        add_to_dict(log_vars_preiteration,evaluation_vals,'test',list(evaluation_vals.keys()))

        if verbose>=2:
            print_var_list=['loss_train','class_loss_train','regularization_loss_train',
                           'loss_test','class_loss_test','regularization_loss_test',
                           'top1_accuracy_train','top5_accuracy_train','top1_accuracy_test','top5_accuracy_test']
            print_var_name_list=["Train loss","Train class loss","Train reg loss","Validation loss","Validation class loss","Validation reg loss","Top1 train accuracy","Top5 train accuracy","Top1 validation accuracy","Top5 validation accuracy"]

            print_eval_metric(log_vars_preiteration, print_var_list, print_var_name_list, 'Pretrain evaluation on all classes')

    # add to history

    history.append(log_vars_preiteration)

    # previous network

    previous_network=tf_networks['network_prev']

    #iteration begins

    for iteration in range(num_iterations):
        #train and test file for each class_batch
        using_classes=class_ord[iteration*fixed_params['class_batch_size']:(iteration+1)*fixed_params['class_batch_size']]
        train_idx=np.array([i in using_classes for i in data_dict_total['Y_train']])
        test_idx=np.array([i in using_classes for i in data_dict_total['Y_test']])
        data_dict=dict(X_train=data_dict_total['X_train'][train_idx],
                       Y_train=data_dict_total['Y_train'][train_idx],
                       sample_weight_train=np.ones_like(np.nonzero(train_idx)[0])*np.nan,
                       X_test=data_dict_total['X_test'][test_idx],
                       Y_test=data_dict_total['Y_test'][test_idx])
        #cumul dataset
        using_classes_cumul=class_ord[:(iteration+1)*fixed_params['class_batch_size']]
        train_idx=np.array([i in using_classes_cumul for i in data_dict_total['Y_train']])
        test_idx=np.array([i in using_classes_cumul for i in data_dict_total['Y_test']])
        data_dict_cumul=dict(X_train=data_dict_total['X_train'][train_idx],
                               Y_train=data_dict_total['Y_train'][train_idx],
                               X_test=data_dict_total['X_test'][test_idx],
                               Y_test=data_dict_total['Y_test'][test_idx])

        #ori dataset
        using_classes_ori=class_ord[:fixed_params['class_batch_size']]
        train_idx=np.array([i in using_classes_ori for i in data_dict_total['Y_train']])
        test_idx=np.array([i in using_classes_ori for i in data_dict_total['Y_test']])
        data_dict_ori=dict(X_train=data_dict_total['X_train'][train_idx],
                               Y_train=data_dict_total['Y_train'][train_idx],
                               X_test=data_dict_total['X_test'][test_idx],
                               Y_test=data_dict_total['Y_test'][test_idx])
        #prev dataset
        using_classes_prev=class_ord[(iteration-1)*fixed_params['class_batch_size']:iteration*fixed_params['class_batch_size']]
        train_idx=np.array([i in using_classes_prev for i in data_dict_total['Y_train']])
        test_idx=np.array([i in using_classes_prev for i in data_dict_total['Y_test']])
        data_dict_prev=dict(X_train=data_dict_total['X_train'][train_idx],
                               Y_train=data_dict_total['Y_train'][train_idx],
                               X_test=data_dict_total['X_test'][test_idx],
                               Y_test=data_dict_total['Y_test'][test_idx])

        if iteration==0 and hyper_params['train_method']=='train_with_sample_weight':
            data_dict['sample_weight_train']=np.ones_like(data_dict['Y_train']).astype(np.float32)
        if iteration>=1 and hyper_params['primary_exemplars'] is not None:
            previous_best_model_params_file=os.path.join(fixed_params['base_dir'],'best_model_params_%d.pkl'%(iteration-1))
            previous_network.load_model(previous_best_model_params_file,session)
            tf_networks['network_train'].load_model(previous_best_model_params_file,session)
            print('Computing fisher information')
            it=iterate_minibatches(data_dict_prev['X_test'],np.arange(len(data_dict_prev['X_test'])),
                                   len(data_dict_prev['X_test']),dataset=dataset)
            X_test_converted=next(it)[0]
            assert len(X_test_converted)==len(data_dict_prev['X_test'])
            fisher_info_val=previous_network.compute_fisher_information(X_test_converted,session)
            print_max(fisher_info_val)
            arr_nan_test(fisher_info_val)
            rv_val=previous_network.get_all_regularizable_variables(session)
            tf_networks['network_train'].set_fisher_variables(fisher_info_val,session)
            tf_networks['network_train'].set_prev_variables(rv_val,session)
            if hyper_params['primary_exemplars']=='icarl_exemplars':
                exemplars_up_to_now,exemplars_up_to_now_label=exemplars_as_training_set(icarl_exemplars,hyper_params['use_fixedsize_exemplars'],class_ord,list(range(0,(iteration)*fixed_params['class_batch_size'])))
            elif hyper_params['primary_exemplars']=='svm_exemplars':
                exemplars_up_to_now,exemplars_up_to_now_label=exemplars_as_training_set(svm_exemplars,hyper_params['use_fixedsize_exemplars'],class_ord,list(range(0,(iteration)*fixed_params['class_batch_size'])))
            assert(np.all(~np.isnan(exemplars_up_to_now)))
            assert(np.all(~np.isnan(exemplars_up_to_now_label)))


            if hyper_params['train_method']=='train_with_sample_weight':
                if type(hyper_params['sample_weight']) in [float,int]:
                    sample_weight=np.ones_like(exemplars_up_to_now_label).astype(np.float32)*hyper_params['sample_weight']
                elif type(hyper_params['sample_weight']) == str:
                    if hyper_params['sample_weight'] == 'half':
                        sample_weight=np.ones_like(exemplars_up_to_now_label).astype(np.float32)*len(data_dict['Y_train'])/len(exemplars_up_to_now_label)
                    elif hyper_params['sample_weight'] == 'balance':
                        counter_train=Counter(data_dict['Y_train'])
                        class_count_train=np.array(list(counter_train.values()))
                        #assert class is balanced in the training set
                        assert np.all(class_count_train==class_count_train[0])
                        counter_exemplars=Counter(exemplars_up_to_now_label)
                        class_count_exemplars=np.array(list(counter_exemplars.values()))
                        assert np.all(class_count_exemplars==class_count_exemplars[0])
                        sample_weight=np.ones_like(exemplars_up_to_now_label)*class_count_train[0]/class_count_exemplars[0]
                    else:
                        assert False
                else:
                    assert False
                data_dict['sample_weight_train']=np.concatenate([np.ones_like(data_dict['Y_train']),
                                                                 sample_weight],axis=0)

                print('Sample weight info:')
                print(Counter(data_dict['sample_weight_train']))

            data_dict['X_train']=np.concatenate([data_dict['X_train'],exemplars_up_to_now],axis=0)
            data_dict['Y_train']=np.concatenate([data_dict['Y_train'],exemplars_up_to_now_label],axis=0)

        # print for verification

        print('Y_train class info:')
        print(Counter(data_dict['Y_train']))

        print('Y_test class info:')
        print(Counter(data_dict['Y_test']))


        #set filename for this iteration

        best_model_params_file=os.path.join(fixed_params['base_dir'],'best_model_params_%d.pkl'%iteration) #save best params after every iteration

        #restore lr at start of iteration

        session.run(tf_variables['lr'].assign(hyper_params['initial_lr']))

        # initialize history at this iteration

        train_counter_var_names=['local_step','global_step']
        loss_var_names=['loss_train','loss_test','best_loss_test','best_loss_test_epoch']
        top1_accuracy_var_names=['top1_accuracy_train','top1_accuracy_test','best_top1_accuracy_test','best_top1_accuracy_test_epoch']
        top5_accuracy_var_names=['top5_accuracy_train','top5_accuracy_test','best_top5_accuracy_test','best_top5_accuracy_test_epoch']
        other_var_names=['epoch_time','mbi']
        log_vars={k:None for k in train_counter_var_names+loss_var_names+top1_accuracy_var_names+top5_accuracy_var_names+other_var_names}
        reset_log_vars(log_vars,list(log_vars.keys()))
        history_iter={}

        #restore model at start of iteration

        if hyper_params['train_method']=='train_cumul':
                print('using cumulative training mode, resetting all parameters before iteration...')
                tf_networks['network_train'].get_variables_initializer().run()

        print('===========Iteration %d============='%(iteration+1))
        print('Using classes %r'%using_classes)

        for epoch in range(num_epochs):

            if epoch in hyper_params['lr_reduction_epoch']:
                new_lr=session.run(tf_variables['lr'].assign(tf_variables['lr']/hyper_params['lr_reduction_rate']))
                print('lr reduced to %f'%new_lr)
            #class batch pretrain evaluation
            if epoch==0 and pretrain_evaluation>=1:
                #set test network
                model_params=tf_networks['network_train'].get_all_model_params(session)
                tf_networks['network_test'].set_model_params(model_params,session)
                tf_networks['network_test'].set_prev_variables(tf_networks['network_train'].get_all_prev_variables(),session)
                tf_networks['network_test'].set_fisher_variables(tf_networks['network_train'].get_all_fisher_variables(),session)
                #plain method evaluation
                #evaluation on trainset
                evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                        class_loss=tf_tensors['class_loss_test'],
                                        regularization_loss=tf_tensors['regularization_loss_test'],
                                        top1_accuracy=tf_tensors['top1_accuracy_test'],
                                        top5_accuracy=tf_tensors['top5_accuracy_test'],
                                        X=tf_tensors['X_test'],
                                        Y=tf_tensors['Y_test'],
                                        fc=tf_networks['network_test'].tf_tensors['fc'])
                evaluation_vals=test_accuracy_evaluation_plain(data_dict['X_train'],
                                                               data_dict['Y_train'],
                                                               hyper_params['test_batch_size'],
                                                               evaluation_tensors,session,dataset=dataset)
                history_var_list=['loss','class_loss','regularization_loss','top1_accuracy','top5_accuracy']
                add_to_dict(log_vars,evaluation_vals,'train',history_var_list)
                #evaluation on testset

                evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                        class_loss=tf_tensors['class_loss_test'],
                                        regularization_loss=tf_tensors['regularization_loss_test'],
                                        top1_accuracy=tf_tensors['top1_accuracy_test'],
                                        top5_accuracy=tf_tensors['top5_accuracy_test'],
                                        X=tf_tensors['X_test'],
                                        Y=tf_tensors['Y_test'],
                                        fc=tf_networks['network_test'].tf_tensors['fc'])

                evaluation_vals=test_accuracy_evaluation_plain(data_dict['X_test'],
                                                               data_dict['Y_test'],
                                                               hyper_params['test_batch_size'],
                                                               evaluation_tensors,session,dataset=dataset)

                history_var_list=['loss','class_loss','regularization_loss','top1_accuracy','top5_accuracy']
                add_to_dict(log_vars,evaluation_vals,'test',history_var_list)

                add_to_list(history_iter,log_vars,'',list(log_vars.keys()))

                if verbose>=2:
                    print_var_list=['loss_train','class_loss_train','regularization_loss_train',
                                   'loss_test','class_loss_test','regularization_loss_test',
                                   'top1_accuracy_train','top5_accuracy_train','top1_accuracy_test','top5_accuracy_test']
                    print_var_name_list=["Train loss","Train class loss","Train reg loss","Validation loss","Validation class loss","Validation reg loss","Top1 train accuracy","Top5 train accuracy","Top1 validation accuracy","Top5 validation accuracy"]
                    print_eval_metric(log_vars,print_var_list,print_var_name_list,'Class batch pretrain evaluation (plain method)')
                    print(evaluation_vals['confusion_matrix'])

            #training
            t=time.time()
            train_tensors=dict((k,tf_tensors.get(k)) for k in ['optimizer_train','loss_train','class_loss_train',
                                                           'regularization_loss_train','X_train','Y_one_hot_train',
                                                          'Y_train','top1_accuracy_train','sample_weight_train',
                                                           'top5_accuracy_train','network_prev_sigmoid'])
            assert np.all(~np.isnan(data_dict['X_train']))
            assert np.all(~np.isnan(data_dict['Y_train']))
            #using distillation
            if hyper_params['train_method']=='train_distillation_and_ground_truth':
                if iteration==0:

                    evaluation_vals_train=train_distillation_and_ground_truth(data_dict['X_train'],data_dict['Y_train'],
                                                                        iteration,hyper_params,fixed_params,train_tensors,session,
                                                                        previous_network=None,class_ord=class_ord,dataset=dataset)

                if iteration>0:
                    evaluation_vals_train=train_distillation_and_ground_truth(data_dict['X_train'],data_dict['Y_train'],
                                                       iteration,hyper_params,fixed_params,train_tensors,
                                                       session,previous_network=previous_network,class_ord=class_ord,dataset=dataset)
            elif hyper_params['train_method']=='train_with_sample_weight':
                evaluation_vals_train=train_with_sample_weight(data_dict['X_train'],data_dict['Y_train'],
                                                 data_dict['sample_weight_train'],
                                                 hyper_params['train_batch_size'],
                                                 train_tensors,
                                                 session,dataset=dataset)

            #without distillation
            elif hyper_params['train_method']=='train_plain':

                evaluation_vals_train=train_plain(data_dict['X_train'],data_dict['Y_train'],hyper_params['train_batch_size'],
                                            train_tensors,session,dataset=dataset)
            elif hyper_params['train_method']=='train_cumul':
                if iteration==0:
                    evaluation_vals_train=train_plain(data_dict_cumul['X_train'],data_dict_cumul['Y_train'],hyper_params['train_batch_size'],
                                            train_tensors,session,dataset=dataset)
                elif iteration>0:
                    evaluation_vals_train=train_plain(data_dict_cumul['X_train'],data_dict_cumul['Y_train'],hyper_params['train_batch_size'],
                                            train_tensors,session,dataset=dataset)
            elif hyper_params['train_method']=='train_distillation':
                if iteration==0:
                    evaluation_vals_train=train_distillation(data_dict['X_train'],data_dict['Y_train'],iteration,hyper_params,
                                                       fixed_params,train_tensors,session,previous_network=None,class_ord=class_ord,dataset=dataset)



                if iteration>0:
                    evaluation_vals_train=train_distillation(data_dict['X_train'],data_dict['Y_train'],
                                                       iteration,hyper_params,fixed_params,train_tensors,
                                                       session,previous_network=previous_network,class_ord=class_ord,dataset=dataset)

            history_var_list=['loss','class_loss','regularization_loss','top1_accuracy','top5_accuracy']
            add_to_dict(log_vars,evaluation_vals_train,'train',history_var_list)
            #get model parameters from train net
            model_params=tf_networks['network_train'].get_all_model_params(session)
            tf_networks['network_test'].set_model_params(model_params,session)
            tf_networks['network_test'].set_prev_variables(tf_networks['network_train'].get_all_prev_variables(),session)
            tf_networks['network_test'].set_fisher_variables(tf_networks['network_train'].get_all_fisher_variables(),session)

            #evaluation on testset
            evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                    class_loss=tf_tensors['class_loss_test'],
                                    regularization_loss=tf_tensors['regularization_loss_test'],
                                    top1_accuracy=tf_tensors['top1_accuracy_test'],
                                    top5_accuracy=tf_tensors['top5_accuracy_test'],
                                    X=tf_tensors['X_test'],
                                    Y=tf_tensors['Y_test'],
                                    fc=tf_networks['network_test'].tf_tensors['fc'])
            evaluation_vals_test=test_accuracy_evaluation_plain(data_dict['X_test'],
                                                           data_dict['Y_test'],
                                                           hyper_params['test_batch_size'],
                                                           evaluation_tensors,session,dataset=dataset)
            history_var_list=['loss','class_loss','regularization_loss','top1_accuracy','top5_accuracy']
            add_to_dict(log_vars,evaluation_vals_test,'test',history_var_list)
            history_var_list=['epoch_time','mbi']
            mbi=session.run(tf_tensors['mbi_op'])
            epoch_time=time.time()-t
            add_to_dict(log_vars,dict(mbi=mbi,epoch_time=epoch_time),None,history_var_list)
            # global step and local step
            log_vars['local_step']+=evaluation_vals_train['local_step']
            log_vars['global_step']+=1

            if evaluation_metric=='top1_accuracy':
                best_metric_val=log_vars['best_top1_accuracy_test']
            elif evaluation_metric=='top5_accuracy':
                best_metric_val=log_vars['best_top5_accuracy_test']
            elif evaluation_metric=='validation_loss':
                best_metric_val=-log_vars['loss_test']
            else:
                raise ValueError('Unknown evaluation metric: '+evaluation_metric)


            if verbose>=2 and epoch%print_freq==0:

                print_var_list=['loss_train','class_loss_train','regularization_loss_train',
                               'loss_test','class_loss_test','regularization_loss_test',
                               'top1_accuracy_train','top5_accuracy_train','top1_accuracy_test','top5_accuracy_test']
                print_var_name_list=["Train loss","Train class loss","Train reg loss","Validation loss","Validation class loss","Validation reg loss","Top1 train accuracy","Top5 train accuracy","Top1 validation accuracy","Top5 validation accuracy"]
                print_eval_metric(log_vars,print_var_list,print_var_name_list,'Epoch %d'%(epoch+1))

                print(evaluation_vals_test['confusion_matrix'])
                nan_test(tf_networks['network_train'])
                
                print('mbi=%d'%(mbi))
                print('time= %f'%(epoch_time))


            if log_vars['loss_test']<log_vars['best_loss_test']:
                log_vars['best_loss_test']=log_vars['loss_test']
                log_vars['best_loss_test_epoch']=epoch+1
            if log_vars['top1_accuracy_test']>log_vars['best_top1_accuracy_test']:
                log_vars['best_top1_accuracy_test']=log_vars['top1_accuracy_test']
                log_vars['best_top1_accuracy_test_epoch']=epoch+1
            if log_vars['top5_accuracy_test']>log_vars['best_top5_accuracy_test']:
                log_vars['best_top5_accuracy_test']=log_vars['top5_accuracy_test']
                log_vars['best_top5_accuracy_test_epoch']=epoch+1

            convert_numpy_dtype(log_vars)

            add_to_list(history_iter,log_vars,'',list(history_iter.keys()))

            if evaluation_metric=='top1_accuracy':
                new_metric_val=log_vars['top1_accuracy_test']
            elif evaluation_metric=='top5_accuracy':
                new_metric_val=log_vars['top5_accuracy_test']
            elif evaluation_metric=='validation_loss':
                new_metric_val=-log_vars['loss_test']

            if new_metric_val>best_metric_val or epoch==0 or (epoch==num_epochs-1):
                if epoch==num_epochs-1:
                    print('saving last-epoch parameters...')
                else:
                    print('saving model parameters...')
                tf_networks['network_train'].save_model(best_model_params_file,session)


        # reload test model
        final_best_model_params_file=os.path.join(fixed_params['base_dir'],'best_model_params_%d.pkl'%(iteration))
        tf_networks['network_test'].load_model(final_best_model_params_file,session)
        tf_networks['network_test'].reset_ewc_variables(session) # should not have ewc at final evaluation time

        # sum up epoch time to iteration time
        history_iter['iteration_time']=sum(history_iter['epoch_time'][1:])

        #plain evaluation before retrain
        plain_evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                class_loss=tf_tensors['class_loss_test'],
                                regularization_loss=tf_tensors['regularization_loss_test'],
                                top1_accuracy=tf_tensors['top1_accuracy_test'],
                                top5_accuracy=tf_tensors['top5_accuracy_test'],
                                X=tf_tensors['X_test'],
                                Y=tf_tensors['Y_test'],
                                fc=tf_networks['network_test'].tf_tensors['fc'])

        eval_vals_test=test_accuracy_evaluation_plain(data_dict['X_test'],
                                                       data_dict['Y_test'],
                                                       hyper_params['test_batch_size'],
                                                       plain_evaluation_tensors,session,dataset=dataset)
        eval_vals_cumul=test_accuracy_evaluation_plain(data_dict_cumul['X_test'],
                                                       data_dict_cumul['Y_test'],
                                                       hyper_params['test_batch_size'],
                                                       plain_evaluation_tensors,session,dataset=dataset)
        eval_vals_ori=test_accuracy_evaluation_plain(data_dict_ori['X_test'],
                                                       data_dict_ori['Y_test'],
                                                       hyper_params['test_batch_size'],
                                                       plain_evaluation_tensors,session,dataset=dataset)

        history_iter.update(dict(best_plain_before_test=eval_vals_test,
                                       best_plain_before_cumul=eval_vals_cumul,
                                       best_plain_before_ori=eval_vals_ori))


        if verbose>=1:
            print('On iteration %d'%(iteration+1))
            print('Plain evaluation before retrain')
            print('\tBest top1 validation accuracy: %f'%eval_vals_test['top1_accuracy'])
            print('\tBest top5 validation accuracy: %f'%eval_vals_test['top5_accuracy'])
            print('\tBest top1 cumul accuracy: %f'%eval_vals_cumul['top1_accuracy'])
            print('\tBest top5 cumul accuracy: %f'%eval_vals_cumul['top5_accuracy'])
            print('\tBest top1 ori accuracy: %f'%eval_vals_ori['top1_accuracy'])
            print('\tBest top5 ori accuracy: %f'%eval_vals_ori['top5_accuracy'])
            print('\tReport string of cumul dataset')
            print(eval_vals_cumul['report_string'])

        #manage classes_up_to_now
        classes_ind_up_to_now=list(range(0,(iteration+1)*fixed_params['class_batch_size']))

        #update exemplars
        feature_map_tensors=dict(X=tf_tensors['X_test'],feature_map=tf_tensors['feature_map_test'])
        if hyper_params['use_fixedsize_exemplars']:
            icarl_exemplars=get_fixedsize_exemplars_set_icarl(data_dict_cumul,hyper_params['exemplars_set_size'],hyper_params['test_batch_size'],feature_map_tensors,class_ord,classes_ind_up_to_now,session,dataset=dataset)
            svm_exemplars,_=get_fixed_size_exemplars_set_by_svm_without_repeat(data_dict_cumul,hyper_params['exemplars_set_size'],
                                                            hyper_params['test_batch_size'],feature_map_tensors,class_ord,classes_ind_up_to_now,session,dataset=dataset)
        else:
            update_exemplars_set_icarl(icarl_exemplars,data_dict_cumul,class_ord,classes_ind_up_to_now,
                                      hyper_params['test_batch_size'],hyper_params['exemplars_set_size'],
                                      feature_map_tensors,session,dataset=dataset)

            svm_exemplars,_=get_fixed_size_exemplars_set_by_svm_without_repeat(data_dict_cumul,
                                                                    hyper_params['exemplars_set_size']*len(classes_ind_up_to_now),
                                                                            hyper_params['test_batch_size'],feature_map_tensors,
                                                                            class_ord,classes_ind_up_to_now,session,dataset=dataset)
        with open(icarl_exemplars_file,'wb') as f:
            pickle.dump(icarl_exemplars,f)
        with open(svm_exemplars_file,'wb') as f:
            pickle.dump(svm_exemplars,f)

        update_exemplars_mean_icarl(icarl_exemplars_mean,icarl_exemplars,hyper_params['test_batch_size'],feature_map_tensors,hyper_params['use_fixedsize_exemplars'],class_ord,classes_ind_up_to_now,session=session,dataset=dataset)
        with open(icarl_exemplars_mean_file,'wb') as f:
            pickle.dump(icarl_exemplars_mean,f)
        #manage primary exemplars
        if hyper_params['primary_exemplars']=='icarl_exemplars':
            exemplars=icarl_exemplars
        elif hyper_params['primary_exemplars']=='svm_exemplars':
            exemplars=svm_exemplars
        elif hyper_params['primary_exemplars'] is None:
            pass
        else:
            assert False
        #retrain last layer
        print('retraining last layer...')
        train_tensors=dict((k,tf_tensors.get(k)) for k in ['loss_train','class_loss_train','regularization_loss_train','optimizer_fc_train','X_train','Y_one_hot_train','Y_train','top1_accuracy_train','sample_weight_train','top5_accuracy_train','network_prev_sigmoid'])
        if hyper_params['final_train_epochs']>0 and hyper_params['primary_exemplars'] is not None:
            tf_networks['network_train'].load_model(final_best_model_params_file)
            last_layer_retrain_with_exemplars(exemplars,class_ord, classes_ind_up_to_now,
                                              hyper_params['train_batch_size'],
                                              train_tensors,tf_networks['network_train'],
                                              final_best_model_params_file,hyper_params['final_train_epochs'],hyper_params['train_method'],hyper_params['use_fixedsize_exemplars'],session=session,dataset=dataset)
            #plain evaluation after retrain
            train_params_last_layer=tf_networks['network_train'].get_model_params(['fc/W','fc/b'],session)
            tf_networks['network_test'].set_model_params(train_params_last_layer)
            plain_evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                                    class_loss=tf_tensors['class_loss_test'],
                                    regularization_loss=tf_tensors['regularization_loss_test'],
                                    top1_accuracy=tf_tensors['top1_accuracy_test'],
                                    top5_accuracy=tf_tensors['top5_accuracy_test'],
                                    X=tf_tensors['X_test'],
                                    Y=tf_tensors['Y_test'],
                                    fc=tf_networks['network_test'].tf_tensors['fc'])
            eval_vals_test=test_accuracy_evaluation_plain(data_dict['X_test'],
                                                           data_dict['Y_test'],
                                                           hyper_params['test_batch_size'],
                                                           plain_evaluation_tensors,session,dataset=dataset)
            eval_vals_cumul=test_accuracy_evaluation_plain(data_dict_cumul['X_test'],
                                                           data_dict_cumul['Y_test'],
                                                           hyper_params['test_batch_size'],
                                                           plain_evaluation_tensors,session,dataset=dataset)
            eval_vals_ori=test_accuracy_evaluation_plain(data_dict_ori['X_test'],
                                                           data_dict_ori['Y_test'],
                                                           hyper_params['test_batch_size'],
                                                           plain_evaluation_tensors,session,dataset=dataset)

            history_iter.update(dict(best_plain_after_test=eval_vals_test,
                                           best_plain_after_cumul=eval_vals_cumul,
                                           best_plain_after_ori=eval_vals_ori))


            if verbose>=1:
                print('Plain evaluation after retrain')
                print('\tBest top1 validation accuracy: %f'%eval_vals_test['top1_accuracy'])
                print('\tBest top5 validation accuracy: %f'%eval_vals_test['top5_accuracy'])
                print('\tBest top1 cumul accuracy: %f'%eval_vals_cumul['top1_accuracy'])
                print('\tBest top5 cumul accuracy: %f'%eval_vals_cumul['top5_accuracy'])
                print('\tBest top1 ori accuracy: %f'%eval_vals_ori['top1_accuracy'])
                print('\tBest top5 ori accuracy: %f'%eval_vals_ori['top5_accuracy'])
                print('\tReport string of cumul dataset')
                print(eval_vals_cumul['report_string'])
        #svm evaluation
        if hyper_params['primary_exemplars'] is not None:
            if hyper_params['use_fixedsize_exemplars']:
                assert np.all(~np.isnan(exemplars))
            else:
                assert(np.all(~np.isnan(exemplars[classes_ind_up_to_now,...])))
            svm_model=svm_retrain_with_exemplars(exemplars,class_ord,classes_ind_up_to_now,
                                           hyper_params['train_batch_size'],feature_map_tensors,hyper_params['use_fixedsize_exemplars'],session,dataset=dataset)
            eval_vals_test=test_evaluation_svm(data_dict['X_test'],data_dict['Y_test'],
                                               hyper_params['test_batch_size'],feature_map_tensors,
                                               svm_model,session=session,dataset=dataset)
            eval_vals_cumul=test_evaluation_svm(data_dict_cumul['X_test'],data_dict_cumul['Y_test'],
                                                hyper_params['test_batch_size'],feature_map_tensors,
                                               svm_model,session=session,dataset=dataset)
            eval_vals_ori=test_evaluation_svm(data_dict_ori['X_test'],data_dict_ori['Y_test'],
                                              hyper_params['test_batch_size'],feature_map_tensors,
                                               svm_model,session=session,dataset=dataset)
            history_iter.update(dict(best_svm_test=eval_vals_test,
                                           best_svm_cumul=eval_vals_cumul,
                                           best_svm_ori=eval_vals_ori))

            if verbose>=1:
                print('SVM evaluation')
                print('\tBest top1 validation accuracy: %f'%eval_vals_test['top1_accuracy'])
                print('\tBest top1 cumul accuracy: %f'%eval_vals_cumul['top1_accuracy'])
                print('\tBest top1 ori accuracy: %f'%eval_vals_ori['top1_accuracy'])
                print('\tReport string of cumul dataset')
                print(eval_vals_cumul['report_string'])

        #exemplars mean evaluation
        assert(np.all(~np.isnan(icarl_exemplars_mean[classes_ind_up_to_now,...])))
        eval_vals_test=test_accuracy_evaluation_ncm(data_dict['X_test'],data_dict['Y_test'],
                                               icarl_exemplars_mean,hyper_params['test_batch_size'],feature_map_tensors,session,dataset=dataset)
        eval_vals_cumul=test_accuracy_evaluation_ncm(data_dict_cumul['X_test'],data_dict_cumul['Y_test'],
                                               icarl_exemplars_mean,hyper_params['test_batch_size'],feature_map_tensors,session,dataset=dataset)
        eval_vals_ori=test_accuracy_evaluation_ncm(data_dict_ori['X_test'],data_dict_ori['Y_test'],
                                               icarl_exemplars_mean,hyper_params['test_batch_size'],feature_map_tensors,session,dataset=dataset)

        history_iter.update(dict(best_exemplars_mean_test=eval_vals_test,
                                       best_exemplars_mean_cumul=eval_vals_cumul,
                                       best_exemplars_mean_ori=eval_vals_ori))
        if verbose>=1:
            print('exemplars mean evaluation')
            print('\tBest top1 validation accuracy: %f'%eval_vals_test['top1_accuracy'])
            print('\tBest top5 validation accuracy: %f'%eval_vals_test['top5_accuracy'])
            print('\tBest top1 cumul accuracy: %f'%eval_vals_cumul['top1_accuracy'])
            print('\tBest top5 cumul accuracy: %f'%eval_vals_cumul['top5_accuracy'])
            print('\tBest top1 ori accuracy: %f'%eval_vals_ori['top1_accuracy'])
            print('\tBest top5 ori accuracy: %f'%eval_vals_ori['top5_accuracy'])
            print('\tReport string of cumul dataset')
            print(eval_vals_cumul['report_string'])
        #update theoretical_mean
        if fixed_params['use_theoretical_mean']:
            theoretical_mean=update_theoretical_mean_icarl(theoretical_mean,data_dict_cumul,hyper_params['test_batch_size'],
                                       feature_map_tensors,class_ord,classes_ind_up_to_now,session,dataset=dataset)
            with open(theoretical_mean_file,'wb') as f:
                pickle.dump(theoretical_mean,f)

            #theoretical mean evaluation

            assert(np.all(~np.isnan(theoretical_mean[classes_ind_up_to_now,...])))
            eval_vals_test=test_accuracy_evaluation_ncm(data_dict['X_test'],data_dict['Y_test'],
                                                   theoretical_mean,hyper_params['test_batch_size'],feature_map_tensors,session,dataset=dataset)
            eval_vals_cumul=test_accuracy_evaluation_ncm(data_dict_cumul['X_test'],data_dict_cumul['Y_test'],
                                                   theoretical_mean,hyper_params['test_batch_size'],feature_map_tensors,session,dataset=dataset)
            eval_vals_ori=test_accuracy_evaluation_ncm(data_dict_ori['X_test'],data_dict_ori['Y_test'],
                                               theoretical_mean,hyper_params['test_batch_size'],feature_map_tensors,session,dataset=dataset)
            history_iter.update(dict(best_theoretical_mean_test=eval_vals_test,
                                           best_theoretical_mean_cumul=eval_vals_cumul,
                                           best_theoretical_mean_ori=eval_vals_ori))
            if verbose>=1:
                print('Theoretical mean evaluation')
                print('\tBest top1 validation accuracy: %f'%eval_vals_test['top1_accuracy'])
                print('\tBest top5 validation accuracy: %f'%eval_vals_test['top5_accuracy'])
                print('\tBest top1 cumul accuracy: %f'%eval_vals_cumul['top1_accuracy'])
                print('\tBest top5 cumul accuracy: %f'%eval_vals_cumul['top5_accuracy'])
                print('\tBest top1 ori accuracy: %f'%eval_vals_ori['top1_accuracy'])
                print('\tBest top5 ori accuracy: %f'%eval_vals_ori['top5_accuracy'])
                print('\tReport string of cumul dataset')
                print(eval_vals_cumul['report_string'])


            # add to history
            history.append(history_iter)

            # save history
            with open(history_file,'wb') as f:
                pickle.dump(history,f)

    #Final evaluation
    print('===========Final Evaluation=============')
    classes_ind_up_to_now=list(range(0,(num_iterations)*fixed_params['class_batch_size']))
    # initialize log_vars_postiteration
    log_vars_postiteration={}

    final_best_model_params_file=os.path.join(fixed_params['base_dir'],'best_model_params_%d.pkl'%(num_iterations-1))
    tf_networks['network_test'].load_model(final_best_model_params_file,session)
    tf_networks['network_test'].reset_ewc_variables(session)
    #plain evaluation
    #evaluation on trainset
    plain_evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                            class_loss=tf_tensors['class_loss_test'],
                            regularization_loss=tf_tensors['regularization_loss_test'],
                            top1_accuracy=tf_tensors['top1_accuracy_test'],
                            top5_accuracy=tf_tensors['top5_accuracy_test'],
                            X=tf_tensors['X_test'],
                            Y=tf_tensors['Y_test'],
                            fc=tf_networks['network_test'].tf_tensors['fc'])
    evaluation_vals=test_accuracy_evaluation_plain(data_dict_total['X_train'],
                                                   data_dict_total['Y_train'],
                                                   hyper_params['test_batch_size'],
                                                   plain_evaluation_tensors,session,dataset=dataset)

    add_to_dict(log_vars_postiteration,evaluation_vals,'plain_train_final',['top1_accuracy','top5_accuracy'])


    #evaluation on testset
    plain_evaluation_tensors=dict(loss=tf_tensors['loss_test'],
                            class_loss=tf_tensors['class_loss_test'],
                            regularization_loss=tf_tensors['regularization_loss_test'],
                            top1_accuracy=tf_tensors['top1_accuracy_test'],
                            top5_accuracy=tf_tensors['top5_accuracy_test'],
                            X=tf_tensors['X_test'],
                            Y=tf_tensors['Y_test'],
                            fc=tf_networks['network_test'].tf_tensors['fc'])
    evaluation_vals=test_accuracy_evaluation_plain(data_dict_total['X_test'],
                                                   data_dict_total['Y_test'],
                                                   hyper_params['test_batch_size'],
                                                   plain_evaluation_tensors,session,dataset=dataset)

    add_to_dict(log_vars_postiteration,evaluation_vals,'plain_test_final',['top1_accuracy','top5_accuracy'])

    if verbose>=2:
        print_var_list=['top1_accuracy_plain_train_final','top5_accuracy_plain_train_final','top1_accuracy_plain_test_final','top5_accuracy_plain_test_final']
        print_var_names_list=["Top1 train accuracy","Top5 train accuracy","Top1 validation accuracy","Top5 validation accuracy"]
        print_eval_metric(log_vars_postiteration,print_var_list,print_var_names_list,'Final evaluation on all classes (plain method)')

    #exemplars mean evaluation
    assert(np.all(~np.isnan(icarl_exemplars_mean[classes_ind_up_to_now,...])))
    feature_map_tensors=dict(X=tf_tensors['X_test'],feature_map=tf_tensors['feature_map_test'])
    eval_vals=test_accuracy_evaluation_ncm(data_dict_total['X_test'],data_dict_total['Y_test'],
                                           icarl_exemplars_mean,hyper_params['test_batch_size'],feature_map_tensors,session,dataset=dataset)
    add_to_dict(log_vars_postiteration,eval_vals,'exemplars_mean_test_final',['top1_accuracy','top5_accuracy'])

    if verbose>=2:
        print('Final evaluation on all classes (exemplars mean)')
        print('\tBest top1 validation accuracy: %f'%eval_vals['top1_accuracy'])
        print('\tBest top5 validation accuracy: %f'%eval_vals['top5_accuracy'])

    #theoretical mean evaluation
    if fixed_params['use_theoretical_mean']:
        assert(np.all(~np.isnan(theoretical_mean[classes_ind_up_to_now,...])))
        feature_map_tensors=dict(X=tf_tensors['X_test'],feature_map=tf_tensors['feature_map_test'])
        eval_vals=test_accuracy_evaluation_ncm(data_dict_total['X_test'],data_dict_total['Y_test'],
                                               theoretical_mean,hyper_params['test_batch_size'],feature_map_tensors,session,dataset=dataset)
        add_to_dict(log_vars_postiteration,eval_vals,'theoretical_mean_test_final',['top1_accuracy','top5_accuracy'])

        if verbose>=2:
            print('Final evaluation on all classes (Theoretical mean)')
            print('\tBest top1 validation accuracy: %f'%eval_vals['top1_accuracy'])
            print('\tBest top5 validation accuracy: %f'%eval_vals['top5_accuracy'])

        # add to history
        history.append(log_vars_postiteration)

        # save history
        with open(history_file,'wb') as f:
            pickle.dump(history,f)

def fit_simplified(tf_tensors,tf_variables,tf_networks,fixed_params,hyper_params,data_dict,session,num_epochs=40,reset=False):
    if reset:
        tf.global_variables_initializer().run()
    train_tensors=dict((k,tf_tensors.get(k)) for k in ['optimizer_train','loss_train','class_loss_train',
                                                           'regularization_loss_train','X_train','Y_one_hot_train',
                                                          'Y_train','top1_accuracy_train','sample_weight_train',
                                                           'top5_accuracy_train','network_prev_sigmoid'])
    test_tensors=dict(loss=tf_tensors['loss_test'],
                        class_loss=tf_tensors['class_loss_test'],
                        regularization_loss=tf_tensors['regularization_loss_test'],
                        top1_accuracy=tf_tensors['top1_accuracy_test'],
                        top5_accuracy=tf_tensors['top5_accuracy_test'],
                        X=tf_tensors['X_test'],
                        Y=tf_tensors['Y_test'],
                        fc=tf_networks['network_test'].tf_tensors['fc'])
    for epoch in range(num_epochs):
        eval_vals_train=train_plain(data_dict['X_train'],data_dict['Y_train'],hyper_params['train_batch_size'],
                                            train_tensors,session,dataset='cifar100')
        model_params=tf_networks['network_train'].get_all_model_params(session)
        model_params_prev=tf_networks['network_test'].get_all_model_params(session)
        tf_networks['network_test'].set_model_params(model_params,session)

        eval_vals_test=test_accuracy_evaluation_plain(data_dict['X_test'],data_dict['Y_test'],
                                                       hyper_params['test_batch_size'],
                                                       test_tensors,session,dataset='cifar100')
        print("Epoch %d:"%(epoch+1))
        print("loss train: %f"%(eval_vals_train['loss_train']))
        print("class loss train: %f"%(eval_vals_train['class_loss_train']))
        print("reg loss train: %f"%(eval_vals_train['regularization_loss_train']))
        print("top1 acc train: %f"%(eval_vals_train['top1_accuracy_train']))
        print("top5 acc train: %f"%(eval_vals_train['top5_accuracy_train']))

        print("loss test: %f"%(eval_vals_test['loss']))
        print("class loss test: %f"%(eval_vals_test['class_loss']))
        print("reg loss test: %f"%(eval_vals_test['regularization_loss']))
        print("top1 acc test: %f"%(eval_vals_test['top1_accuracy']))
        print("top5 acc test: %f"%(eval_vals_test['top5_accuracy']))
        eval_vals_train2=test_accuracy_evaluation_plain(data_dict['X_train'],data_dict['Y_train'],
                                                       hyper_params['test_batch_size'],
                                                       test_tensors,session,dataset='cifar100')
        print(eval_vals_train2['confusion_matrix'])
        for k in model_params.keys():
            avg_change=np.mean(np.abs(model_params[k]-model_params_prev[k]))
            print('%s: %f'%(k,avg_change))





