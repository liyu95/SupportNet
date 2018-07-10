# SupportNet
SupportNet: a novel incremental learning framework through deep learning and support data

This repository shows the implementation of SupportNet, solving the catastrophic forgetting problem efficiently and effectively. SupportNet combines the strength of deep learning and support vector machine (SVM), where SVM is used to identify the support data from the old data, which are fed to the deep learning model together with the new data for further training so that the model can review the essential information of the old data when learning the new information. Two powerful consolidation regularizers are applied to ensure the robustness of the learned model. Comprehensive experiments on various tasks, including enzyme function prediction, subcellular structure classification and breast tumor classification, show that SupportNet drastically outperforms the state-of-the-art incremental learning methods and reaches similar performance as the deep learning model trained from scratch on both old and new data.

## Paper
https://arxiv.org/abs/1806.02942

## Datasets
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

### For HeLa and BreakHis Datasets
The code the result are in the submodule *myIL*. It's written using Jupyter Notebook. Every code and result were thus recorded. 



