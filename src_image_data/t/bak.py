def fit_one_class(tf_tensors,tf_variables,tf_networks,fixed_params,hyper_params,data_dict,session,resume=False,
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
        tf_log_var_val['loss_train']=0
        tf_log_var_val['class_loss_train']=0
        tf_log_var_val['regularization_loss_train']=0
        tf_log_var_val['top1_accuracy_train']=0
        tf_log_var_val['top5_accuracy_train']=0
        num_batches_train=int((np.sum(train_idx)/hyper_params['train_batch_size']))
        model_params=tf_networks['train_network'].get_all_model_params(session)
        tf_networks['train_network'].set_model_params(model_params,session)

        for X_minibatch,Y_minibatch in iterate_minibatches(data_dict['X_train'][train_idx],data_dict['Y_train'][train_idx],hyper_params['train_batch_size'],shuffle=False):
            local_loss,local_class_loss,local_regularization_loss,local_top1_accuracy_train,local_top5_accuracy_train=\
            session.run([tf_tensors['loss_train'],tf_tensors['class_loss_train'],tf_tensors['regularization_loss_train'],tf_tensors['top1_accuracy_train'],tf_tensors['top5_accuracy_train']],
                        feed_dict={tf_tensors['X_train']:X_minibatch,
                                   tf_tensors['Y_train']:Y_minibatch})

            tf_log_var_val['loss_train']+=local_loss/num_batches_train
            tf_log_var_val['class_loss_train']+=local_class_loss/num_batches_train
            tf_log_var_val['regularization_loss_train']+=local_regularization_loss/num_batches_train
            tf_log_var_val['top1_accuracy_train']+=local_top1_accuracy_train/num_batches_train
            tf_log_var_val['top5_accuracy_train']+=local_top5_accuracy_train/num_batches_train
            
        tf_log_var_val['loss_test']=0
        tf_log_var_val['class_loss_test']=0
        tf_log_var_val['regularization_loss_test']=0
        tf_log_var_val['top1_accuracy_test']=0
        tf_log_var_val['top5_accuracy_test']=0
        num_batches_test=int((np.sum(test_idx)/hyper_params['test_batch_size']))
        model_params=tf_networks['test_network'].get_all_model_params(session)
        tf_networks['test_network'].set_model_params(model_params,session)

        for X_minibatch,Y_minibatch in iterate_minibatches(data_dict['X_test'][test_idx],data_dict['Y_test'][test_idx],hyper_params['test_batch_size'],shuffle=False):
            local_loss,local_class_loss,local_regularization_loss,local_top1_accuracy_test,local_top5_accuracy_test=\
            session.run([tf_tensors['loss_test'],tf_tensors['class_loss_test'],tf_tensors['regularization_loss_test'],tf_tensors['top1_accuracy_test'],tf_tensors['top5_accuracy_test']],
                        feed_dict={tf_tensors['X_test']:X_minibatch,
                                   tf_tensors['Y_test']:Y_minibatch})

            tf_log_var_val['loss_test']+=local_loss/num_batches_test
            tf_log_var_val['class_loss_test']+=local_class_loss/num_batches_test
            tf_log_var_val['regularization_loss_test']+=local_regularization_loss/num_batches_test
            tf_log_var_val['top1_accuracy_test']+=local_top1_accuracy_test/num_batches_test
            tf_log_var_val['top5_accuracy_test']+=local_top5_accuracy_test/num_batches_test
            
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
        
        tf_log_var_val['loss_test']=0
        tf_log_var_val['class_loss_test']=0
        tf_log_var_val['regularization_loss_test']=0
        tf_log_var_val['top1_accuracy_test']=0
        tf_log_var_val['top5_accuracy_test']=0
        num_batches_test=int((np.sum(test_idx)/hyper_params['test_batch_size']))
        model_params=tf_networks['train_network'].get_all_model_params(session)
        tf_networks['test_network'].set_model_params(model_params,session)
        for X_minibatch,Y_minibatch in iterate_minibatches(data_dict['X_test'][test_idx],data_dict['Y_test'][test_idx],hyper_params['test_batch_size'],shuffle=False):
            local_loss,local_class_loss,local_regularization_loss,local_top1_accuracy_test,local_top5_accuracy_test=\
            session.run([tf_tensors['loss_test'],tf_tensors['class_loss_test'],tf_tensors['regularization_loss_test'],tf_tensors['top1_accuracy_test'],tf_tensors['top5_accuracy_test']],
                        feed_dict={tf_tensors['X_test']:X_minibatch,
                                   tf_tensors['Y_test']:Y_minibatch})

            tf_log_var_val['loss_test']+=local_loss/num_batches_test
            tf_log_var_val['class_loss_test']+=local_class_loss/num_batches_test
            tf_log_var_val['regularization_loss_test']+=local_regularization_loss/num_batches_test
            tf_log_var_val['top1_accuracy_test']+=local_top1_accuracy_test/num_batches_test
            tf_log_var_val['top5_accuracy_test']+=local_top5_accuracy_test/num_batches_test
            
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