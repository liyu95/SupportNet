from scipy.misc import imresize
import numpy as np
def data_dict_resize(data_dict):
    n_train=len(data_dict['X_train'])
    n_test=len(data_dict['X_test'])
    X_train=np.zeros([n_train,32,32,1])
    X_test=np.zeros([n_test,32,32,1])
    for i in range(n_train):
        X_train[i,:,:,0]=imresize(data_dict['X_train'][i].reshape([28,28]),[32,32])
    for i in range(n_test):
        X_test[i,:,:,0]=imresize(data_dict['X_test'][i].reshape([28,28]),[32,32])
    return dict(X_train=X_train,Y_train=data_dict['Y_train'].astype(np.int32),X_test=X_test,Y_test=data_dict['Y_test'])


