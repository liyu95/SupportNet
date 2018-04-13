# SupportNet
SupportNet: a novel incremental learning framework through deep learning and support data

## Datasets
1. The EC dataset: http://www.cbrc.kaust.edu.sa/DEEPre/dataset.html
2. The HeLa dataset: http://murphylab.web.cmu.edu/data/2DHeLa
3. The BreakHis dataset: https://web.inf.ufpr.br/vri/breast-cancer-database/

## Prerequisite
1. Tensorflow (https://www.tensorflow.org/)
2. TFLearn (tflearn.org/)
3. CUDA (https://developer.nvidia.com/cuda-downloads)
4. cuDNN (https://developer.nvidia.com/cudnn)
5. sklearn (scikit-learn.org/)
6. numpy (www.numpy.org/)
7. Jupyter notebook (jupyter.org/)

## Source Code and Experimental records
### For EC number dataset
The code is in folder *src_ec*. The whole program can be run by execute *main.sh*. That file could take advantage of *supportnet.py*, which is the complete implementation of SupportNet. *icarl_level_1.py* shows our implementation of iCaRL on this specific dataset. The other files are some temp files for testing or libraries.
The experimental results were recorded in *level_1_result.md*

### For HeLa and BreakHis datasets
The code the result are recorded in the submodule *myIL*. It's written using Jupyter Notebook. Every code and result were thus recorded. 



