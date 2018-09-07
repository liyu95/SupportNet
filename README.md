# SupportNet
SupportNet: a novel incremental learning framework through deep learning and support data

This repository shows the implementation of SupportNet, solving the catastrophic forgetting problem efficiently and effectively. 
A plain well-trained deep learning model often does not have the ability to learn new knowledge without forgetting the previously learned knowledge, which is known as *catastrophic forgetting*. Here we propose a novel method, SupportNet, to efficiently and effectively solve the catastrophic forgetting problem in the class incremental learning scenario. SupportNet combines the strength of deep learning and support vector machine (SVM), where SVM is used to identify the support data from the old data, which are fed to the deep learning model together with the new data for further training so that the model can review the essential information of the old data when learning the new information. Two powerful consolidation regularizers are applied to stabilize the learned representation and ensure the robustness of the learned model. We validate our method both theoretically and also empirically with comprehensive experiments on various tasks, which shows that SupportNet drastically outperforms the state-of-the-art incremental learning methods and even reaches similar performance as the deep learning model trained from scratch on both old and new data.

## Paper
https://arxiv.org/abs/1806.02942

## Datasets
1. MNIST
2. CIFAR-10 and CIFAR-100
1. The EC dataset: http://www.cbrc.kaust.edu.sa/DEEPre/dataset.html (**This file contains the orignal sequence data and the labels. The pickle files, which are preprocessed from this original sequence data and are the feature files ready for usage in the script can be provided based on request. We are sorry that we cannot completely release them currently, since this paper has not been officiaully published. Those feature files would be released after this paper been published.**)
2. The HeLa dataset: http://murphylab.web.cmu.edu/data/2DHeLa
3. The BreakHis dataset: https://web.inf.ufpr.br/vri/breast-cancer-database/

## Prerequisites
1. Tensorflow (https://www.tensorflow.org/)
2. TFLearn (tflearn.org/)
3. CUDA (https://developer.nvidia.com/cuda-downloads)
4. cuDNN (https://developer.nvidia.com/cudnn)
5. sklearn (scikit-learn.org/)
6. numpy (www.numpy.org/)
7. Jupyter notebook (jupyter.org/)

## Source Code and Experimental Records
### For EC number dataset
The code is in folder *src_ec*. The whole program can be run by execute *main.sh*. That file could take advantage of *supportnet.py*, which is the complete implementation of SupportNet. *icarl_level_1.py* shows our implementation of iCaRL on this specific dataset. The other files are some temp files for testing or libraries.

The experimental results were recorded in *level_1_result.md*.

### For CIFAR-10, CIFAR-100, HeLa and BreakHis Datasets
The code the result are in the submodule *myIL*. It's written using Jupyter Notebook. Every code and result were thus recorded. 

## Incremental Learning
<p align="center">
<img src="https://github.com/lykaust15/SupportNet/blob/master/result/incremental_learning.png" alt="Incremental Learning" width="400"/>
</p>

Illustration of class incremental learning. After we train a base model using all the available data at a certain time point (e.g., classes $1-N_1$), new data belonging to new classes may continuously appear (e.g., classes $N_2-N_3$, classes $N_4-N_5$, etc) and we need to equip the model with the ability to handle the new classes.

## Catastrophic Forgetting
<p align="center">
<img src="https://github.com/lykaust15/SupportNet/blob/master/result/cm.png" alt="Catastrophic Forgetting" width="800"/>
</p>
The confusion matrix of incrementally training a deep learning model following the class incremental learning scenario using different methods. (A) Random guess, (B) fine-tune (only fine tune the model with the newest data), (C) iCarl, (D) SupportNet. (B) illustrates the problem of catastrophic forgetting. If we only use the newest data to further train the model, the model does not have the ability to handle the old classes anymore.

## Main framework
<p align="center">
<img src="https://github.com/lykaust15/SupportNet/blob/master/result/framework.png" width="400"/>
</p>

Overview of our framework. The basic idea is to incrementally train a deep learning model efficiently using the new class data and the support data of the old classes. We divide the deep learning model into two parts, the mapping function (all the layers before the last layer) and the softmax layer (the last layer). Using the learned representation produced by the mapping function, we train an SVM, with which we can find the support vector index and thus the support data of old classes. To stabilize the learned representation of old data, we apply two novel consolidation regularizers to the network.

## Main result
<p align="center">
<img src="https://github.com/lykaust15/SupportNet/blob/master/result/main_result.png" width="700"/>
</p>

Main results. (A)-(E): Performance comparison between SupportNet and five competing methods on the five datasets in terms of accuracy. For the SupportNet and iCaRL methods, we set the support data (examplar) size as 2000 for MNIST, CIFAR-10 and enzyme data, 80 for the HeLa dataset, and 1600 for the breast tumor dataset. (F): The accuracy deviation of SupportNet from the ''All Data'' method with respect to the size of the support data. The x-axis shows the support data size. The y-axis is the test accuracy deviation of SupportNet from the ''All Data'' method after incrementally learning all the classes of the HeLa subcellular structure dataset.

**Notice that** for the MNIST data, we can reach almost the same performance as using all data during each incremental learning iteration. That is, with our framework, we only need 2000 data points to reach the same performance as using 50,000 data points on that specific data.
