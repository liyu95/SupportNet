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
from tensorflow.python.ops.variable_scope import VariableScope
from train_utils import assert_session

def relu(x, name, alpha):
    if alpha > 0:
        return tf.maximum(alpha * x, x, name=name)
    else:
        return tf.nn.relu(x, name=name)


def get_variable(name, shape, dtype, initializer, trainable=True, regularizer=None):
    var = tf.get_variable(name, shape=shape, dtype=dtype,
                          initializer=initializer, regularizer=regularizer, trainable=trainable,
                          collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    return var

def conv(inp, name_or_scope, size, out_channels, strides=[1, 1, 1, 1],
         dilation=None, padding='SAME', apply_relu=True, alpha=0.0,bias=True,
         initializer=tf.contrib.layers.xavier_initializer_conv2d(),reuse=False):

    batch_size = inp.get_shape().as_list()[0]
    res1 = inp.get_shape().as_list()[1]
    res2 = inp.get_shape().as_list()[1]
    in_channels = inp.get_shape().as_list()[3]
    def conv_inner():
        W = get_variable("W", shape=[size, size, in_channels, out_channels], dtype=tf.float32,
                initializer=initializer, regularizer=tf.nn.l2_loss)#lizx: use this will add this variable to tf.GraphKeys.REGULARIZATION_LOSSES
        b = get_variable("b", shape=[1, 1, 1, out_channels], dtype=tf.float32,
                         initializer=tf.zeros_initializer(),trainable=bias)
        if dilation:
            assert(strides == [1, 1, 1, 1])
            out = tf.add(tf.nn.atrous_conv2d(inp, W, rate=dilation, padding=padding), b, name='convolution')
            out.set_shape([batch_size, res1, res2, out_channels])
        else:
            out = tf.add(tf.nn.conv2d(inp, W, strides=strides, padding=padding), b, name='convolution')

        if apply_relu:
            out = relu(out, alpha=alpha, name='relu')
        return out

    with tf.variable_scope(name_or_scope,reuse=reuse):
        out=conv_inner()
    return out

def batch_norm(inp, name_or_scope, phase, decay=0.9,reuse=False):#lizx: good example of batchnorm

    channels = inp.get_shape().as_list()[3]
    def batch_norm_inner():
        moving_mean = get_variable("mean", shape=[channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
        moving_variance = get_variable("var", shape=[channels], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)

        offset = get_variable("offset", shape=[channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        scale = get_variable("scale", shape=[channels], dtype=tf.float32, initializer=tf.constant_initializer(1.0), regularizer=tf.nn.l2_loss)

        mean, variance = tf.nn.moments(inp, axes=[0, 1, 2], shift=moving_mean) #lizx: tf.nn.moments has random behaviour


        mean_op = moving_mean.assign(decay * moving_mean + (1 - decay) * mean)
        var_op = moving_variance.assign(decay * moving_variance + (1 - decay) * variance)

        assert(phase in ['train', 'test'])
        if phase == 'train':
            with tf.control_dependencies([mean_op, var_op]):
                return tf.nn.batch_normalization(inp, mean, variance, offset, scale, 0.01, name='norm')
        else:
            return tf.nn.batch_normalization(inp, moving_mean, moving_variance, offset, scale, 0.01, name='norm') #lizx: during inference time the moving_mean and moving_variance are fixed variables
    with tf.variable_scope(name_or_scope,reuse=reuse):
        batch_norm_out=batch_norm_inner()
    return batch_norm_out


def softmax(target, axis, name_or_scope,reuse=False):
    with tf.variable_scope(name_or_scope,reuse=reuse):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target - max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / normalize
        return softmax

def pool(inp, name_or_scope, kind, size, stride, padding='SAME',reuse=False):

    assert kind in ['max', 'avg']

    strides = [1, stride, stride, 1]
    sizes = [1, size, size, 1]
    def pool_inner():
        if kind == 'max':
            out = tf.nn.max_pool(inp, sizes, strides=strides, padding=padding, name=kind)
        else:
            out = tf.nn.avg_pool(inp, sizes, strides=strides, padding=padding, name=kind)
        return out
    with tf.variable_scope(name_or_scope,reuse=reuse):
        pool_out=pool_inner()
    return pool_out
def squeeze_excitation_layer(input_x, out_dim, ratio, name_or_scope,reuse=False):
    with tf.variable_scope(name_or_scope,reuse=reuse) :
        squeeze = tflearn.layers.conv.global_avg_pool(input_x,name='Global_avg_pooling')
        excitation = tflearn.layers.core.fully_connected(squeeze,n_units=out_dim / ratio, bias=False, name='fc1',regularizer='L2')
        excitation = tf.nn.relu(excitation)
        excitation = tflearn.layers.core.fully_connected(excitation,n_units=out_dim, bias=False, name='fc2',regularizer='L2')
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation

        return scale
def get_variable_assign_ops_and_phs(tf_variables):
    tf_assign_ops={}
    tf_assign_phs={}
    for k,v in tf_variables.items():
        id_tensor=tf.identity(v)
        tf_assign_phs[k]=tf.placeholder(dtype=id_tensor.dtype,shape=id_tensor.shape.as_list())
        tf_assign_ops[k]=v.assign(tf_assign_phs[k])
    return tf_assign_ops,tf_assign_phs
def assign_to_variables(tf_assign_ops,tf_assign_phs,value_dict,sess=None):

    sess=assert_session(sess)
    assign_ops=[]
    feed_dict={}
    for k,v in value_dict.items():
        assign_ops.append(tf_assign_ops[k])
        feed_dict[tf_assign_phs[k]]=v
    sess.run(assign_ops,feed_dict=feed_dict)
class ResNet(object):
    def residual_block(inp, phase, alpha=0.0,nom='a',increase_dim=False,last=False,projection=True,se=False):
        input_num_filters = inp.get_shape().as_list()[3]
        if increase_dim:
            first_stride = [1, 2, 2, 1]
            out_num_filters = input_num_filters*2
        else:
            first_stride = [1, 1, 1, 1]
            out_num_filters = input_num_filters

        layer = conv(inp, 'resconv1_'+nom, size=3, strides=first_stride, out_channels=out_num_filters, alpha=alpha, padding='SAME',bias=False,apply_relu=False,initializer=tf.contrib.keras.initializers.he_normal()) #lizx: add bias=False,apply_relu=False
        layer = batch_norm(layer, 'batch_norm_resconv1_'+nom, phase=phase)
        layer= relu(layer,name='relu',alpha=alpha) #lizx: add relu after batch_norm
        layer = conv(layer, 'resconv2_'+nom, size=3, strides=[1, 1, 1, 1], out_channels=out_num_filters, apply_relu=False,alpha=alpha, padding='SAME',bias=False,initializer=tf.contrib.keras.initializers.he_normal()) #lizx: add bias=False
        layer = batch_norm(layer, 'batch_norm_resconv2_'+nom, phase=phase)
        #layer=relu(layer,name='relu',alpha=alpha) #lizx:should not have relu here, because relu is applied after adding with shortcut path
        if se:
            layer = squeeze_excitation_layer(layer, out_num_filters, ratio=4, name_or_scope='squeeze_excitation_'+nom)
        if increase_dim:
            if projection:
                projection = conv(inp, 'projconv_'+nom, size=1, strides=[1, 2, 2, 1], out_channels=out_num_filters, alpha=alpha, apply_relu=False,padding='SAME',bias=False)
                projection = batch_norm(projection, 'batch_norm_projconv_'+nom, phase=phase)
                if last:
                    block = layer + projection
                else:
                    block = layer + projection
                    block = tf.nn.relu(block, name='relu')
            else:
                identity = inp[:,::2,::2,:]
                padding = tf.pad(identity,[[0,0],[0,0],[0,0],[out_num_filters//4,out_num_filters//4]])
                if last:
                    block = layer + padding
                else:
                    block = layer + padding
                    block = tf.nn.relu(block,name='relu')
        else:
            if last:
                block = layer + inp
            else:
                block = layer + inp
                block = tf.nn.relu(block, name='relu')

        return block
    def __init__(self,resnet_type,phase,input_tensor=None,num_outputs=100,alpha=0.0,name='',use_fisher=False,se=False,input_channels=3):
        self.tf_tensors={}
        self.tf_variables={}
        self.graph=tf.get_default_graph()
        self.num_outputs=num_outputs
        if name=='':
            name=resnet_type
        if input_tensor is not None:
            self.tf_tensors['input']=input_tensor
        if resnet_type=='ResNet18':
            with tf.variable_scope(name) as net_scope:
                if 'input' not in self.tf_tensors:
                    self.tf_tensors['input']=tf.placeholder(tf.float32,shape=[None,224,224,input_channels])
                layer =self.tf_tensors['input']
                layer = conv(layer,"conv_1",size=7,strides=[1, 2, 2, 1], out_channels=64, alpha=alpha, padding='SAME',bias=False,apply_relu=False) #lizx: bias=False,apply_relu=False
                self.tf_tensors['conv_1']=layer
                layer = batch_norm(layer, 'batch_norm_1', phase=phase)
                self.tf_tensors['batch_norm_1']=layer
                layer = relu(layer,name='relu_1',alpha=alpha) #lizx: add relu after batch_norm
                layer = pool(layer, 'pool1', 'max', size=3, stride=2)
                self.tf_tensors['pool_1']=layer

                # First stack of residual blocks

                for letter in ['2a','2b']:
                    layer = ResNet.residual_block(layer, phase, alpha=0.0,nom=letter,se=se)
                    self.tf_tensors['residual_block_'+letter]=layer

                # Second stack of residual blocks
                layer = ResNet.residual_block(layer, phase, alpha=0.0,nom='3a',increase_dim=True,se=se)
                self.tf_tensors['residual_block_3a']=layer
                for letter in ['3b']:
                    layer = ResNet.residual_block(layer, phase, alpha=0.0,nom=letter,se=se)
                    self.tf_tensors['residual_block_'+letter]=layer

                # Third stack of residual blocks
                layer = ResNet.residual_block(layer, phase, alpha=0.0,nom='4a',increase_dim=True,se=se)
                self.tf_tensors['residual_block_4a']=layer
                for letter in ['4b']:
                    layer = ResNet.residual_block(layer, phase, alpha=0.0,nom=letter,se=se)
                    self.tf_tensors['residual_block_'+letter]=layer

                # Fourth stack of residual blocks
                layer = ResNet.residual_block(layer, phase, alpha=0.0,nom='5a',increase_dim=True,se=se)
                self.tf_tensors['residual_block_5a']=layer
                layer = ResNet.residual_block(layer, phase, alpha=0.0,nom='5b',increase_dim=False,last=True,se=se)
                self.tf_tensors['residual_block_5b']=layer

                layer = pool(layer, 'pool_last', 'avg', size=7, stride=1,padding='VALID')
                self.tf_tensors['pool_last']=layer
                layer = conv(layer, name_or_scope='fc', size=1, out_channels=num_outputs, padding='VALID', apply_relu=False, alpha=alpha)[:, 0, 0,:]
                self.tf_tensors['fc']=layer

        elif resnet_type=='ResNet32':
            print('Use ResNet32')
            with tf.variable_scope(name) as net_scope:
                if 'input' not in self.tf_tensors:
                    #lizx: for_debug
                    self.tf_tensors['input']=tf.placeholder(tf.float32,shape=[None,32,32,input_channels])
#                     self.tf_tensors['input']=tflearn.input_data(shape=[None, 32, 32, 3])
                layer=self.tf_tensors['input']
                layer=conv(layer, name_or_scope='conv_1', size=3, out_channels=16, strides=[1, 1, 1, 1],padding='SAME', apply_relu=False,bias=False,initializer=tf.contrib.keras.initializers.he_normal()) #lizx: change to apply_relu=False, add bias=False
                self.tf_tensors['conv_1']=layer
                layer=batch_norm(layer, name_or_scope='batch_norm_1', phase=phase)
                self.tf_tensors['batch_norm_1']=layer
                layer=relu(layer,name='relu_1',alpha=alpha)
                for i in 'abcde':
                    layer=ResNet.residual_block(layer,phase,nom='2'+i,projection=False,increase_dim=False,se=se)
                    self.tf_tensors['residual_block_2'+i]=layer
                layer=ResNet.residual_block(layer,phase,nom='3a',projection=False,increase_dim=True,se=se)
                self.tf_tensors['residual_block_3a']=layer
                for i in 'bcde':
                    layer=ResNet.residual_block(layer,phase,nom='3'+i,projection=False,increase_dim=False,se=se)
                    self.tf_tensors['residual_block_3'+i]=layer
                layer=ResNet.residual_block(layer,phase,nom='4a',projection=False,increase_dim=True,se=se)
                self.tf_tensors['residual_block_4a']=layer
                for i in 'bcd':
                    layer=ResNet.residual_block(layer,phase,nom='4'+i,projection=False,increase_dim=False,se=se)
                    self.tf_tensors['residual_block_4'+i]=layer
                layer=ResNet.residual_block(layer,phase,nom='4e',projection=False,last=True,increase_dim=False,se=se)
                self.tf_tensors['residual_block_4e']=layer
                layer=pool(layer,'pool_last','avg',size=8,stride=1,padding='VALID')
                self.tf_tensors['pool_last']=layer
                layer=conv(layer,name_or_scope='fc',size=1,out_channels=num_outputs,padding='VALID',apply_relu=False,alpha=alpha)[:,0,0,:]
                self.tf_tensors['fc']=layer

        elif resnet_type=='ResNet32_tflearn':
            print('Use ResNet32_tflearn')
            n=5
            with tf.variable_scope(name) as net_scope:
                if 'input' not in self.tf_tensors:
                    self.tf_tensors['input']=tflearn.input_data(shape=[None, 32, 32, 3])
                net=self.tf_tensors['input']
                net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001,weights_init=tf.contrib.keras.initializers.he_normal(),bias=False)

#                 net = tflearn.conv_2d(net, 16, 3,weights_init=tf.contrib.keras.initializers.he_normal(),bias=False)
#                 net=conv(net, name_or_scope='conv_1', size=3, out_channels=16, strides=[1, 1, 1, 1],padding='SAME', apply_relu=False,bias=False,initializer=tf.contrib.keras.initializers.he_normal()) #lizx: change to apply_relu=False, add bias=False

# #################rewrite conv####################
# #                 W = get_variable("W", shape=[3, 3, 3, 16], dtype=tf.float32,
# #                 initializer=tf.contrib.keras.initializers.he_normal(), regularizer=tf.nn.l2_loss)
# #                 net=tf.nn.conv2d(net, W, strides=[1,1,1,1], padding='SAME')
# #################################################
# #                 self.tf_tensors['conv_1']=net
# #                 net=batch_norm(net, name_or_scope='batch_norm_1', phase=phase)
# #                 self.tf_tensors['batch_norm_1']=net
# #                 net=relu(net,name='relu_1',alpha=alpha)
                net = tflearn.residual_block(net, n, 16)
                net = tflearn.residual_block(net, 1, 32, downsample=True)
                net = tflearn.residual_block(net, n-1, 32)
                net = tflearn.residual_block(net, 1, 64, downsample=True)
                net = tflearn.residual_block(net, n-1, 64)
                net = tflearn.batch_normalization(net)
                net = tflearn.activation(net, 'relu')
                net = tflearn.global_avg_pool(net)
                self.tf_tensors['pool_last']=tf.expand_dims(tf.expand_dims(net,1),1)
                net = tflearn.fully_connected(net, num_outputs, name='fc')
                self.tf_tensors['fc']=net
        elif resnet_type=='ResNet34':
            with tf.variable_scope(name) as net_scope:
                if 'input' not in self.tf_tensors:
                    self.tf_tensors['input']=tf.placeholder(tf.float32,shape=[None,224,224,input_channels])
                layer =self.tf_tensors['input']
                layer = conv(layer,"conv_1",size=7,strides=[1, 2, 2, 1], out_channels=64, alpha=alpha, padding='SAME',bias=False,apply_relu=False) #lizx: bias=False,apply_relu=False
                self.tf_tensors['conv_1']=layer
                layer = batch_norm(layer, 'batch_norm_1', phase=phase)
                self.tf_tensors['batch_norm_1']=layer
                layer = relu(layer,name='relu_1',alpha=alpha) #lizx: add relu after batch_norm
                layer = pool(layer, 'pool1', 'max', size=3, stride=2)
                self.tf_tensors['pool_1']=layer

                # First stack of residual blocks

                for letter in 'abc':
                    layer = ResNet.residual_block(layer, phase, alpha=0.0,nom='2'+letter,se=se)
                    self.tf_tensors['residual_block_2'+letter]=layer

                # Second stack of residual blocks
                layer = ResNet.residual_block(layer, phase, alpha=0.0,nom='3a',increase_dim=True,se=se)
                self.tf_tensors['residual_block_3a']=layer
                for letter in 'bcd':
                    layer = ResNet.residual_block(layer, phase, alpha=0.0,nom='3'+letter,se=se)
                    self.tf_tensors['residual_block_3'+letter]=layer

                # Third stack of residual blocks
                layer = ResNet.residual_block(layer, phase, alpha=0.0,nom='4a',increase_dim=True,se=se)
                self.tf_tensors['residual_block_4a']=layer
                for letter in 'bcdef':
                    layer = ResNet.residual_block(layer, phase, alpha=0.0,nom='4'+letter,se=se)
                    self.tf_tensors['residual_block_4'+letter]=layer

                # Fourth stack of residual blocks
                layer = ResNet.residual_block(layer, phase, alpha=0.0,nom='5a',increase_dim=True,se=se)
                self.tf_tensors['residual_block_5a']=layer
                for letter in 'b':
                    layer = ResNet.residual_block(layer, phase, alpha=0.0,nom='5'+letter,se=se)
                    self.tf_tensors['residual_block_5'+letter]=layer
                layer = ResNet.residual_block(layer, phase, alpha=0.0,nom='5c',increase_dim=False,last=True,se=se)
                self.tf_tensors['residual_block_5b']=layer

                layer = pool(layer, 'pool_last', 'avg', size=7, stride=1,padding='VALID')
                self.tf_tensors['pool_last']=layer
                layer = conv(layer, name_or_scope='fc', size=1, out_channels=num_outputs, padding='VALID', apply_relu=False, alpha=alpha)[:, 0, 0,:]
                self.tf_tensors['fc']=layer
        tf_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=net_scope.name)
        self.tf_variables={v.name.split(':')[0][len(net_scope.name)+1:]: v for v in tf_variables}
        if resnet_type=='ResNet32_tflearn':
            if 'ResidualBlock/BatchNormalization/is_training' in self.tf_variables:
                del self.tf_variables['ResidualBlock/BatchNormalization/is_training']
        self.tf_variables_ops,self.tf_variables_phs=get_variable_assign_ops_and_phs(self.tf_variables)
        self.tf_tensors['l2_loss']=tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope= net_scope.name))
        self.regularizable_variables={v:self.tf_variables[v] for v in self.tf_variables.keys() if v.endswith('W') or v.endswith('scale')}
        self.regularizable_variables_ops,self.regularizable_variables_phs=get_variable_assign_ops_and_phs(self.regularizable_variables)
        trainable_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=net_scope.name)
        self.trainable_variables={v.name.split(':')[0][len(net_scope.name)+1:]: v for v in trainable_variables}
        self.trainable_variables_ops,self.trainable_variables_phs=get_variable_assign_ops_and_phs(self.trainable_variables)
        if use_fisher:
            self.fisher_variables={}
            self.prev_variables={}
            with tf.variable_scope(net_scope):
                for k in self.regularizable_variables.keys():
                    v=self.regularizable_variables[k]
                    self.fisher_variables[k]=\
                    tf.get_variable(k+'_fisher', dtype=tf.float32,initializer=tf.zeros_like(v),trainable=False)
                    self.prev_variables[k]=\
                    tf.get_variable(k+'_prev', dtype=tf.float32,initializer=tf.zeros_like(v),trainable=False)
                self.fisher_variables_ops,self.fisher_variables_phs=get_variable_assign_ops_and_phs(self.fisher_variables)
                self.prev_variables_ops,self.prev_variables_phs=get_variable_assign_ops_and_phs(self.prev_variables)
            cond_log_prob=self.tf_tensors['fc']
            cond_prob=tf.nn.softmax(self.tf_tensors['fc'])
            log_cond_prob=tf.log(cond_prob)
            Y = tf.cast(tf.stop_gradient(tf.multinomial(log_cond_prob, 1))[0,0],tf.int32)
            log_likelihood=tf.log(cond_prob[0,Y])
            rv_keys=list(self.regularizable_variables.keys())
            grads_one_example=tf.gradients(log_likelihood,[self.regularizable_variables[k] for k in rv_keys])
            self.rv_grads=dict(zip(rv_keys,grads_one_example))

            ewc_losses=[]
            for v in self.regularizable_variables.keys():
                loss_one_var=tf.reduce_sum(self.fisher_variables[v]*(self.regularizable_variables[v]-self.prev_variables[v])**2)
                ewc_losses.append(loss_one_var)
            self.tf_tensors['ewc_loss']=tf.reduce_sum(ewc_losses)
        self.net_scope=net_scope
    def get_model_params(self,name_list,sess=None):
        sess=assert_session(sess)
        if isinstance(name_list,str):
            name_list=[name_list]
        elif not isinstance(name_list,(list,tuple)):
            raise TypeError('name_list must be str, list or tuple')
        value_list=sess.run([self.tf_variables[v] for v in name_list])
        return dict(zip(name_list,value_list))

    def get_all_model_params(self,sess=None):
        sess=assert_session(sess)
        return self.get_model_params(list(self.tf_variables.keys()),sess)
    def set_model_params(self,value_dict,sess=None):
        sess=assert_session(sess)
        assign_to_variables(self.tf_variables_ops,self.tf_variables_phs,value_dict,sess)

    def save_model(self,file_name,sess=None):
        sess=assert_session(sess)
        model_params=self.get_all_model_params(sess)
        pickle.dump(model_params,open(file_name,'wb'))

    def load_model(self,file_name,sess=None):
        sess=assert_session(sess)
        model_params=pickle.load(open(file_name,'rb'))
        self.set_model_params(model_params,sess)
    def get_variables_initializer(self):
        variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=self.net_scope.name)
        return tf.variables_initializer(variables)
    def compute_fisher_information(self,X_train,sess=None):
        #may be only applicable when the last layer goes train with softmax function
        sess=assert_session(sess)
        rv_grad_keys=list(self.rv_grads.keys())
        rv_grad_tensors=[self.rv_grads[k] for k in rv_grad_keys]
        F_accum={}
        for v in rv_grad_keys:
            F_accum[v]=np.zeros(self.rv_grads[v].shape.as_list())
        for i in range(len(X_train)):
            rv_grads_val=sess.run(rv_grad_tensors,feed_dict={self.tf_tensors['input']:X_train[i:i+1]})
            for j,v in enumerate(rv_grad_keys):
                F_accum[v]+=rv_grads_val[j]**2
        for k in F_accum.keys():
            F_accum[k]/=len(X_train)
        return F_accum
    def set_prev_variables(self,prev_rv,sess=None):
        sess=assert_session(sess)
        assign_to_variables(self.prev_variables_ops,self.prev_variables_phs,prev_rv,sess)
    def get_all_prev_variables(self,sess=None):
        sess=assert_session(sess)
        keys_list=list(self.prev_variables.keys())
        value_list=sess.run([self.prev_variables[v] for v in keys_list])
        return dict(zip(keys_list,value_list))
    def get_all_regularizable_variables(self,sess=None):
        sess=assert_session(sess)
        return self.get_model_params(list(self.regularizable_variables.keys()),sess)
    def set_fisher_variables(self,fisher_var,sess=None):
        sess=assert_session(sess)
        assign_to_variables(self.fisher_variables_ops,self.fisher_variables_phs,fisher_var,sess)
    def get_all_fisher_variables(self,sess=None):
        sess=assert_session(sess)
        keys_list=list(self.fisher_variables.keys())
        value_list=sess.run([self.fisher_variables[v] for v in keys_list])
        return dict(zip(keys_list,value_list))
    def reset_ewc_variables(self,sess=None):
        sess=assert_session(sess)
        sess.run(tf.variables_initializer(list(self.fisher_variables.values())+list(self.prev_variables.values())))




