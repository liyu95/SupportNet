## baseline performance using all data and the usual training procedure
### 6 class
The confusion matrix is as follows: 

[[302   4  20   8   1   0]
 [  4 822  18   7   0   1]
 [  3  16 568   5   0   0]
 [  5  12  10 120   5   2]
 [  6   3   4   7  98   2]
 [  0   3   1   1   0 162]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.90      0.92       335
          1       0.96      0.96      0.96       852
          2       0.91      0.96      0.94       592
          3       0.81      0.78      0.79       154
          4       0.94      0.82      0.87       120
          5       0.97      0.97      0.97       167

avg / total       0.93      0.93      0.93      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.933333.

The Cohen's Kappa socre is 0.910173.

The micro precistion is 0.933333, the macro precision is 0.922899.

The micro recall is 0.933333, the macro recall is 0.898615.

The micro F1 score is 0.933333, the macro F1 score is 0.909783.

## baseline all incremental learning result
### 2 class
The confusion matrix is as follows: 

[[323  12]
 [  7 845]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.98      0.96      0.97       335
          1       0.99      0.99      0.99       852

avg / total       0.98      0.98      0.98      1187




Here is the evaluation of the model performance: 

The accuracy score is 0.983993.

The Cohen's Kappa socre is 0.960312.

The micro precistion is 0.983993, the macro precision is 0.982393.

The micro recall is 0.983993, the macro recall is 0.977982.

The micro F1 score is 0.983993, the macro F1 score is 0.980155.

### 3 class
The confusion matrix is as follows: 

[[301   4  30]
 [  3 823  26]
 [  3  27 562]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.98      0.90      0.94       335
          1       0.96      0.97      0.96       852
          2       0.91      0.95      0.93       592

avg / total       0.95      0.95      0.95      1779




Here is the evaluation of the model performance: 

The accuracy score is 0.947723.

The Cohen's Kappa socre is 0.915954.

The micro precistion is 0.947723, the macro precision is 0.951180.

The micro recall is 0.947723, the macro recall is 0.937931.

The micro F1 score is 0.947723, the macro F1 score is 0.943817.

### 4 class
The confusion matrix is as follows: 

[[306   3  17   9]
 [  4 807  28  13]
 [  3   7 577   5]
 [  6  15  16 117]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.96      0.91      0.94       335
          1       0.97      0.95      0.96       852
          2       0.90      0.97      0.94       592
          3       0.81      0.76      0.79       154

avg / total       0.94      0.93      0.93      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.934816.

The Cohen's Kappa socre is 0.903384.

The micro precistion is 0.934816, the macro precision is 0.911522.

The micro recall is 0.934816, the macro recall is 0.898755.

The micro F1 score is 0.934816, the macro F1 score is 0.904415.

### 5 class
The confusion matrix is as follows: 

[[303   2  22   7   1]
 [  4 798  20   9  21]
 [  2  10 576   1   3]
 [ 10  13  12 114   5]
 [ 10   7  12   3  88]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.92      0.90      0.91       335
          1       0.96      0.94      0.95       852
          2       0.90      0.97      0.93       592
          3       0.85      0.74      0.79       154
          4       0.75      0.73      0.74       120

avg / total       0.92      0.92      0.91      2053




Here is the evaluation of the model performance: 

The accuracy score is 0.915246.

The Cohen's Kappa socre is 0.880230.

The micro precistion is 0.915246, the macro precision is 0.875225.

The micro recall is 0.915246, the macro recall is 0.857533.

The micro F1 score is 0.915246, the macro F1 score is 0.865247.

### 6 class
The confusion matrix is as follows: 

[[305   1  15   4   4   6]
 [  4 799  26  17   6   0]
 [  3  10 559   2  11   7]
 [  7   8   9 123   4   3]
 [  2   3   4  10  99   2]
 [  2   1   9   1   1 153]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.91      0.93       335
          1       0.97      0.94      0.95       852
          2       0.90      0.94      0.92       592
          3       0.78      0.80      0.79       154
          4       0.79      0.82      0.81       120
          5       0.89      0.92      0.91       167

avg / total       0.92      0.92      0.92      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.918018.

The Cohen's Kappa socre is 0.890323.

The micro precistion is 0.918018, the macro precision is 0.880864.

The micro recall is 0.918018, the macro recall is 0.888728.

The micro F1 score is 0.918018, the macro F1 score is 0.884510.

## only the support data incremental result, should be worse than the support with the ewc
### 2class
The confusion matrix is as follows: 

[[2996   12]
 [ 118 7547]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.96      1.00      0.98      3008
          1       1.00      0.98      0.99      7665

avg / total       0.99      0.99      0.99     10673




Here is the evaluation of the model performance: 

The accuracy score is 0.987820.

The Cohen's Kappa socre is 0.970230.

The micro precistion is 0.987820, the macro precision is 0.980260.

The micro recall is 0.987820, the macro recall is 0.990308.

The micro F1 score is 0.987820, the macro F1 score is 0.985113.

### 3 class
The confusion matrix is as follows: 

[[213   0 122]
 [  1 704 147]
 [  1   6 585]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.99      0.64      0.77       335
          1       0.99      0.83      0.90       852
          2       0.69      0.99      0.81       592

avg / total       0.89      0.84      0.85      1779




Here is the evaluation of the model performance: 

The accuracy score is 0.844295.

The Cohen's Kappa socre is 0.751412.

The micro precistion is 0.844295, the macro precision is 0.889086.

The micro recall is 0.844295, the macro recall is 0.816763.

The micro F1 score is 0.844295, the macro F1 score is 0.828361.


### 4 class
The confusion matrix is as follows: 

[[ 477    0  264  264]
 [   3 1155  102 1296]
 [   0    0 1530  246]
 [   1    1   12  140]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.99      0.47      0.64      1005
          1       1.00      0.45      0.62      2556
          2       0.80      0.86      0.83      1776
          3       0.07      0.91      0.13       154

avg / total       0.91      0.60      0.68      5491




Here is the evaluation of the model performance: 

The accuracy score is 0.601348.

The Cohen's Kappa socre is 0.477959.

The micro precistion is 0.601348, the macro precision is 0.716162.

The micro recall is 0.601348, the macro recall is 0.674271.

The micro F1 score is 0.601348, the macro F1 score is 0.557063.


## the support with ewc result, sample: 100, lam: 1
### 2 class
The confusion matrix is as follows: 

[[326   9]
 [ 16 836]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.95      0.97      0.96       335
          1       0.99      0.98      0.99       852

avg / total       0.98      0.98      0.98      1187




Here is the evaluation of the model performance: 

The accuracy score is 0.978939.

The Cohen's Kappa socre is 0.948343.

The micro precistion is 0.978939, the macro precision is 0.971283.

The micro recall is 0.978939, the macro recall is 0.977177.

The micro F1 score is 0.978939, the macro F1 score is 0.974170.

### 3 class
The confusion matrix is as follows: 

[[ 486    0  184]
 [   2 1436  266]
 [   1    8  583]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.99      0.73      0.84       670
          1       0.99      0.84      0.91      1704
          2       0.56      0.98      0.72       592

avg / total       0.91      0.84      0.86      2966




Here is the evaluation of the model performance: 

The accuracy score is 0.844572.

The Cohen's Kappa socre is 0.746670.

The micro precistion is 0.844572, the macro precision is 0.850900.

The micro recall is 0.844572, the macro recall is 0.850964.

The micro F1 score is 0.844572, the macro F1 score is 0.822839.


### 4 class
The confusion matrix is as follows: 

[[121   0  83 131]
 [  0 468  15 369]
 [  0   0 470 122]
 [  1   1   4 148]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.99      0.36      0.53       335
          1       1.00      0.55      0.71       852
          2       0.82      0.79      0.81       592
          3       0.19      0.96      0.32       154

avg / total       0.88      0.62      0.68      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.624418.

The Cohen's Kappa socre is 0.505656.

The micro precistion is 0.624418, the macro precision is 0.750889.

The micro recall is 0.624418, the macro recall is 0.666362.

The micro F1 score is 0.624418, the macro F1 score is 0.591500.




## interesting result, even test on the whole training set, the learn model is underfited, which indicate the representation is still need to be improved.

### test data
The confusion matrix is as follows: 

[[110   0  66 159]
 [  0 406  14 432]
 [  0   0 501  91]
 [  0   2   4 148]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       1.00      0.33      0.49       335
          1       1.00      0.48      0.64       852
          2       0.86      0.85      0.85       592
          3       0.18      0.96      0.30       154

avg / total       0.89      0.60      0.65      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.602690.

The Cohen's Kappa socre is 0.484154.

The micro precistion is 0.602690, the macro precision is 0.757455.

The micro recall is 0.602690, the macro recall is 0.653052.

The micro F1 score is 0.602690, the macro F1 score is 0.572739.



### training data
The confusion matrix is as follows: 

[[ 998    2  715 1293]
 [   4 3620  117 3924]
 [   0    0 4615  710]
 [   0   12   16 1350]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       1.00      0.33      0.50      3008
          1       1.00      0.47      0.64      7665
          2       0.84      0.87      0.86      5325
          3       0.19      0.98      0.31      1378

avg / total       0.89      0.61      0.66     17376




Here is the evaluation of the model performance: 

The accuracy score is 0.609058.

The Cohen's Kappa socre is 0.491093.

The micro precistion is 0.609058, the macro precision is 0.755611.

The micro recall is 0.609058, the macro recall is 0.662601.

The micro F1 score is 0.609058, the macro F1 score is 0.576515.


## build the support data from all the previous data, 0 lambda, based on loss function
### 2 class
The confusion matrix is as follows: 

[[330   5]
 [ 22 830]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.99      0.96       335
          1       0.99      0.97      0.98       852

avg / total       0.98      0.98      0.98      1187




Here is the evaluation of the model performance: 

The accuracy score is 0.977254.

The Cohen's Kappa socre is 0.944708.

The micro precistion is 0.977254, the macro precision is 0.965756.

The micro recall is 0.977254, the macro recall is 0.979627.

The micro F1 score is 0.977254, the macro F1 score is 0.972347.

### 3 class
The confusion matrix is as follows: 

[[ 460    2  208]
 [   2 1536  166]
 [   2   17  573]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.99      0.69      0.81       670
          1       0.99      0.90      0.94      1704
          2       0.61      0.97      0.74       592

avg / total       0.91      0.87      0.87      2966




Here is the evaluation of the model performance: 

The accuracy score is 0.866150.

The Cohen's Kappa socre is 0.776816.

The micro precistion is 0.866150, the macro precision is 0.861410.

The micro recall is 0.866150, the macro recall is 0.851960.

The micro F1 score is 0.866150, the macro F1 score is 0.832849.

### 4 class
The confusion matrix is as follows: 

[[ 424    2   78  166]
 [   2 1218   34  450]
 [   0   10  708  466]
 [   1    4    8  141]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.99      0.63      0.77       670
          1       0.99      0.71      0.83      1704
          2       0.86      0.60      0.70      1184
          3       0.12      0.92      0.20       154

avg / total       0.91      0.67      0.75      3712




Here is the evaluation of the model performance: 

The accuracy score is 0.671067.

The Cohen's Kappa socre is 0.556583.

The micro precistion is 0.671067, the macro precision is 0.737593.

The micro recall is 0.671067, the macro recall is 0.715295.

The micro F1 score is 0.671067, the macro F1 score is 0.627681.

### 5 class
The confusion matrix is as follows: 

[[141   0  71   6 117]
 [  0 556  16 146 134]
 [  0   6 413  17 156]
 [  1   3  11  79  60]
 [  0   0   6   2 112]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.99      0.42      0.59       335
          1       0.98      0.65      0.78       852
          2       0.80      0.70      0.74       592
          3       0.32      0.51      0.39       154
          4       0.19      0.93      0.32       120

avg / total       0.84      0.63      0.68      2053




Here is the evaluation of the model performance: 

The accuracy score is 0.633707.

The Cohen's Kappa socre is 0.528135.

The micro precistion is 0.633707, the macro precision is 0.657061.

The micro recall is 0.633707, the macro recall is 0.643487.

The micro F1 score is 0.633707, the macro F1 score is 0.566463.


### 6 class
The confusion matrix is as follows: 

[[ 444    2    4    4   22  194]
 [   4 1186   12  170    6  326]
 [   2   10  718   40   10  404]
 [  10   10    4  178   16   90]
 [   8    4    0    4  172   52]
 [   1    1    0    0    0  165]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.95      0.66      0.78       670
          1       0.98      0.70      0.81      1704
          2       0.97      0.61      0.75      1184
          3       0.45      0.58      0.51       308
          4       0.76      0.72      0.74       240
          5       0.13      0.99      0.24       167

avg / total       0.89      0.67      0.74      4273




Here is the evaluation of the model performance: 

The accuracy score is 0.670021.

The Cohen's Kappa socre is 0.587948.

The micro precistion is 0.670021, the macro precision is 0.706988.

The micro recall is 0.670021, the macro recall is 0.707955.

The micro F1 score is 0.670021, the macro F1 score is 0.636644.


## build the support data from all the previous data, lambda: 10, based on loss function
### 5 class
### 6 class
The confusion matrix is as follows: 

[[224   2   4  80  15  10]
 [  5 625  13 115   3  91]
 [  1  18 362 177   3  31]
 [  9   6   4  97  16  22]
 [  2   0   0   7  99  12]
 [  1   0   0   1   0 165]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.93      0.67      0.78       335
          1       0.96      0.73      0.83       852
          2       0.95      0.61      0.74       592
          3       0.20      0.63      0.31       154
          4       0.73      0.82      0.77       120
          5       0.50      0.99      0.66       167

avg / total       0.85      0.71      0.75      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.708108.

The Cohen's Kappa socre is 0.633103.

The micro precistion is 0.708108, the macro precision is 0.710106.

The micro recall is 0.708108, the macro recall is 0.742768.

The micro F1 score is 0.708108, the macro F1 score is 0.682367.


## build the support data from all the previous data, lambda: 10, based on SVM
### 2 class
The confusion matrix is as follows: 

[[2996   12]
 [ 118 7547]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.96      1.00      0.98      3008
          1       1.00      0.98      0.99      7665

avg / total       0.99      0.99      0.99     10673




Here is the evaluation of the model performance: 

The accuracy score is 0.987820.

The Cohen's Kappa socre is 0.970230.

The micro precistion is 0.987820, the macro precision is 0.980260.

The micro recall is 0.987820, the macro recall is 0.990308.

The micro F1 score is 0.987820, the macro F1 score is 0.985113.

### 3 class
The confusion matrix is as follows: 

[[309   8  18]
 [  6 824  22]
 [  3  14 575]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.97      0.92      0.95       335
          1       0.97      0.97      0.97       852
          2       0.93      0.97      0.95       592

avg / total       0.96      0.96      0.96      1779




Here is the evaluation of the model performance: 

The accuracy score is 0.960090.

The Cohen's Kappa socre is 0.935995.

The micro precistion is 0.960090, the macro precision is 0.960218.

The micro recall is 0.960090, the macro recall is 0.953603.

The micro F1 score is 0.960090, the macro F1 score is 0.956577.

### 4 class, Total nSV of the first 3 class: 658
The confusion matrix is as follows: 

[[287   3   6  39]
 [  5 767  24  56]
 [  3  15 536  38]
 [ 10   4   5 135]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.86      0.90       335
          1       0.97      0.90      0.93       852
          2       0.94      0.91      0.92       592
          3       0.50      0.88      0.64       154

avg / total       0.92      0.89      0.90      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.892395.

The Cohen's Kappa socre is 0.844329.

The micro precistion is 0.892395, the macro precision is 0.838884.

The micro recall is 0.892395, the macro recall is 0.884745.

The micro F1 score is 0.892395, the macro F1 score is 0.848309.

### 5 class, Total nSV of the first 4 class: 1499
The confusion matrix is as follows: 

[[274   9   2  13  37]
 [  3 756   6   3  84]
 [  1  19 493   7  72]
 [ 11  22   6  74  41]
 [  4   7   0   3 106]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.82      0.87       335
          1       0.93      0.89      0.91       852
          2       0.97      0.83      0.90       592
          3       0.74      0.48      0.58       154
          4       0.31      0.88      0.46       120

avg / total       0.89      0.83      0.85      2053




Here is the evaluation of the model performance: 

The accuracy score is 0.829518.

The Cohen's Kappa socre is 0.765764.

The micro precistion is 0.829518, the macro precision is 0.777839.

The micro recall is 0.829518, the macro recall is 0.780371.

The micro F1 score is 0.829518, the macro F1 score is 0.744289.

### 6 class, Total nSV of the first 5 class: 1770
The confusion matrix is as follows: 

[[303   2   1   6   6  17]
 [  3 720   2  12  26  89]
 [  2  13 494  13  32  38]
 [ 28  14   0  71   2  39]
 [  5   9   3   3  75  25]
 [  1   2   0   1   0 163]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.89      0.90      0.90       335
          1       0.95      0.85      0.89       852
          2       0.99      0.83      0.90       592
          3       0.67      0.46      0.55       154
          4       0.53      0.62      0.57       120
          5       0.44      0.98      0.61       167

avg / total       0.87      0.82      0.83      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.822523.

The Cohen's Kappa socre is 0.768303.

The micro precistion is 0.822523, the macro precision is 0.743735.

The micro recall is 0.822523, the macro recall is 0.774349.

The micro F1 score is 0.822523, the macro F1 score is 0.736667.

## build the support data from all the previous data, lambda: 1000, based on SVM
### 2 class
The confusion matrix is as follows: 

[[324  11]
 [ 12 840]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.96      0.97      0.97       335
          1       0.99      0.99      0.99       852

avg / total       0.98      0.98      0.98      1187




Here is the evaluation of the model performance: 

The accuracy score is 0.980623.

The Cohen's Kappa socre is 0.952217.

The micro precistion is 0.980623, the macro precision is 0.975680.

The micro recall is 0.980623, the macro recall is 0.976540.

The micro F1 score is 0.980623, the macro F1 score is 0.976109.

### 3 class, nSV first two class: 193
The confusion matrix is as follows: 

[[293   6  36]
 [  4 814  34]
 [  2   5 585]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.98      0.87      0.92       335
          1       0.99      0.96      0.97       852
          2       0.89      0.99      0.94       592

avg / total       0.95      0.95      0.95      1779




Here is the evaluation of the model performance: 

The accuracy score is 0.951096.

The Cohen's Kappa socre is 0.921595.

The micro precistion is 0.951096, the macro precision is 0.953243.

The micro recall is 0.951096, the macro recall is 0.939401.

The micro F1 score is 0.951096, the macro F1 score is 0.944441.

### 4 class, nSV first 3 class: 822
The confusion matrix is as follows: 

[[288   7   5  35]
 [  4 763  19  66]
 [  4  24 503  61]
 [  6   7   2 139]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.95      0.86      0.90       335
          1       0.95      0.90      0.92       852
          2       0.95      0.85      0.90       592
          3       0.46      0.90      0.61       154

avg / total       0.91      0.88      0.89      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.875841.

The Cohen's Kappa socre is 0.821111.

The micro precistion is 0.875841, the macro precision is 0.829712.

The micro recall is 0.875841, the macro recall is 0.876875.

The micro F1 score is 0.875841, the macro F1 score is 0.833953.

### 5 class, nSV first 4 class: 1588
The confusion matrix is as follows: 

[[291   8   8   5  23]
 [  4 782  20   7  39]
 [  3  11 538   1  39]
 [ 10  15   6  82  41]
 [  6   3   5   2 104]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.93      0.87      0.90       335
          1       0.95      0.92      0.94       852
          2       0.93      0.91      0.92       592
          3       0.85      0.53      0.65       154
          4       0.42      0.87      0.57       120

avg / total       0.90      0.88      0.88      2053




Here is the evaluation of the model performance: 

The accuracy score is 0.875304.

The Cohen's Kappa socre is 0.826304.

The micro precistion is 0.875304, the macro precision is 0.816422.

The micro recall is 0.875304, the macro recall is 0.818883.

The micro F1 score is 0.875304, the macro F1 score is 0.794974.

### 6 class, nSV first 5 class: 1597
The confusion matrix is as follows: 

[[290   7   8  11  17   2]
 [  3 796  12   9  12  20]
 [  1  32 523   6  15  15]
 [  9  19   2 113  10   1]
 [  4   3   7   7  98   1]
 [  1   4   1   0   1 160]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.87      0.90       335
          1       0.92      0.93      0.93       852
          2       0.95      0.88      0.91       592
          3       0.77      0.73      0.75       154
          4       0.64      0.82      0.72       120
          5       0.80      0.96      0.87       167

avg / total       0.90      0.89      0.89      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.891892.

The Cohen's Kappa socre is 0.855617.

The micro precistion is 0.891892, the macro precision is 0.838388.

The micro recall is 0.891892, the macro recall is 0.865318.

The micro F1 score is 0.891892, the macro F1 score is 0.848420.

## build the support data from all the previous data, lambda: 100, based on SVM
### 2 class
The confusion matrix is as follows: 

[[298  37]
 [  2 850]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.99      0.89      0.94       335
          1       0.96      1.00      0.98       852

avg / total       0.97      0.97      0.97      1187




Here is the evaluation of the model performance: 

The accuracy score is 0.967144.

The Cohen's Kappa socre is 0.916249.

The micro precistion is 0.967144, the macro precision is 0.975810.

The micro recall is 0.967144, the macro recall is 0.943602.

The micro F1 score is 0.967144, the macro F1 score is 0.958078.

### 3 class, nsv: 201
The confusion matrix is as follows: 

[[298  19  18]
 [  7 833  12]
 [  7  41 544]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.96      0.89      0.92       335
          1       0.93      0.98      0.95       852
          2       0.95      0.92      0.93       592

avg / total       0.94      0.94      0.94      1779




Here is the evaluation of the model performance: 

The accuracy score is 0.941540.

The Cohen's Kappa socre is 0.905589.

The micro precistion is 0.941540, the macro precision is 0.945225.

The micro recall is 0.941540, the macro recall is 0.928724.

The micro F1 score is 0.941540, the macro F1 score is 0.936336.

### 4 class, nsv: 955
The confusion matrix is as follows: 

[[287   3   2  43]
 [  6 670   5 171]
 [ 21  29 464  78]
 [  2   4   2 146]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.91      0.86      0.88       335
          1       0.95      0.79      0.86       852
          2       0.98      0.78      0.87       592
          3       0.33      0.95      0.49       154

avg / total       0.90      0.81      0.84      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.810657.

The Cohen's Kappa socre is 0.736178.

The micro precistion is 0.810657, the macro precision is 0.792886.

The micro recall is 0.810657, the macro recall is 0.843734.

The micro F1 score is 0.810657, the macro F1 score is 0.776601.

### 5 class, nsv: 1569
The confusion matrix is as follows: 

[[261   4   1   5  64]
 [  2 791   5  10  44]
 [  2  13 489   1  87]
 [  7  17   4  89  37]
 [  3   2   3   2 110]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.95      0.78      0.86       335
          1       0.96      0.93      0.94       852
          2       0.97      0.83      0.89       592
          3       0.83      0.58      0.68       154
          4       0.32      0.92      0.48       120

avg / total       0.91      0.85      0.87      2053




Here is the evaluation of the model performance: 

The accuracy score is 0.847540.

The Cohen's Kappa socre is 0.790235.

The micro precistion is 0.847540, the macro precision is 0.806615.

The micro recall is 0.847540, the macro recall is 0.805622.

The micro F1 score is 0.847540, the macro F1 score is 0.770023.

### 6 class, nsv: 1244
The confusion matrix is as follows: 

[[296   2   1   7  21   8]
 [  3 611   8  26  14 190]
 [  2   8 509   3  36  34]
 [  2   3   2 126  14   7]
 [  4   2   3  13  96   2]
 [  1   1   0   0   3 162]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.96      0.88      0.92       335
          1       0.97      0.72      0.83       852
          2       0.97      0.86      0.91       592
          3       0.72      0.82      0.77       154
          4       0.52      0.80      0.63       120
          5       0.40      0.97      0.57       167

avg / total       0.89      0.81      0.83      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.810811.

The Cohen's Kappa socre is 0.758762.

The micro precistion is 0.810811, the macro precision is 0.758746.

The micro recall is 0.810811, the macro recall is 0.841460.

The micro F1 score is 0.810811, the macro F1 score is 0.770980.

## build the support data from all the previous data, lambda: 10000, based on SVM
### 2 class
The confusion matrix is as follows: 

[[305  30]
 [  6 846]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.98      0.91      0.94       335
          1       0.97      0.99      0.98       852

avg / total       0.97      0.97      0.97      1187




Here is the evaluation of the model performance: 

The accuracy score is 0.969671.

The Cohen's Kappa socre is 0.923479.

The micro precistion is 0.969671, the macro precision is 0.973230.

The micro recall is 0.969671, the macro recall is 0.951703.

The micro F1 score is 0.969671, the macro F1 score is 0.961720.

### 3 class, nsv: 218
The confusion matrix is as follows: 

[[273   8  54]
 [  4 810  38]
 [  4  15 573]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.97      0.81      0.89       335
          1       0.97      0.95      0.96       852
          2       0.86      0.97      0.91       592

avg / total       0.94      0.93      0.93      1779




Here is the evaluation of the model performance: 

The accuracy score is 0.930860.

The Cohen's Kappa socre is 0.888774.

The micro precistion is 0.930860, the macro precision is 0.935191.

The micro recall is 0.930860, the macro recall is 0.911178.

The micro F1 score is 0.930860, the macro F1 score is 0.919827.

### 4 class, nsv: 1314
The confusion matrix is as follows: 

[[280   2  17  36]
 [  2 601  25 224]
 [ 15  23 493  61]
 [  4   3   8 139]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.93      0.84      0.88       335
          1       0.96      0.71      0.81       852
          2       0.91      0.83      0.87       592
          3       0.30      0.90      0.45       154

avg / total       0.88      0.78      0.81      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.782721.

The Cohen's Kappa socre is 0.700139.

The micro precistion is 0.782721, the macro precision is 0.773953.

The micro recall is 0.782721, the macro recall is 0.819147.

The micro F1 score is 0.782721, the macro F1 score is 0.753402.

### 5 class, nsv: 2868
The confusion matrix is as follows: 

[[270   7   8   8  42]
 [  6 779  14  31  22]
 [ 13  26 475   8  70]
 [  4  17   4 117  12]
 [  6   8   1   4 101]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.90      0.81      0.85       335
          1       0.93      0.91      0.92       852
          2       0.95      0.80      0.87       592
          3       0.70      0.76      0.73       154
          4       0.41      0.84      0.55       120

avg / total       0.88      0.85      0.86      2053




Here is the evaluation of the model performance: 

The accuracy score is 0.848514.

The Cohen's Kappa socre is 0.790581.

The micro precistion is 0.848514, the macro precision is 0.777053.

The micro recall is 0.848514, the macro recall is 0.824812.

The micro F1 score is 0.848514, the macro F1 score is 0.783933.


### 6 class, nsv: 1612
The confusion matrix is as follows: 

[[291   4   9   3  16  12]
 [  5 734   7  17  15  74]
 [ 10  15 435   9  22 101]
 [  3  14   1 108  10  18]
 [  2   6   0   1 104   7]
 [  1   1   0   1   1 163]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.93      0.87      0.90       335
          1       0.95      0.86      0.90       852
          2       0.96      0.73      0.83       592
          3       0.78      0.70      0.74       154
          4       0.62      0.87      0.72       120
          5       0.43      0.98      0.60       167

avg / total       0.88      0.83      0.84      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.826577.

The Cohen's Kappa socre is 0.774643.

The micro precistion is 0.826577, the macro precision is 0.779016.

The micro recall is 0.826577, the macro recall is 0.834828.

The micro F1 score is 0.826577, the macro F1 score is 0.782766.

## build the support data from all the previous data, lambda: 0.01, based on SVM
### 2 class
The confusion matrix is as follows: 

[[315  20]
 [  8 844]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.98      0.94      0.96       335
          1       0.98      0.99      0.98       852

avg / total       0.98      0.98      0.98      1187




Here is the evaluation of the model performance: 

The accuracy score is 0.976411.

The Cohen's Kappa socre is 0.941137.

The micro precistion is 0.976411, the macro precision is 0.976042.

The micro recall is 0.976411, the macro recall is 0.965454.

The micro F1 score is 0.976411, the macro F1 score is 0.970565.

### 3 class, nsv: 183
The confusion matrix is as follows: 

[[290   3  42]
 [ 10 808  34]
 [  2  10 580]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.96      0.87      0.91       335
          1       0.98      0.95      0.97       852
          2       0.88      0.98      0.93       592

avg / total       0.95      0.94      0.94      1779




Here is the evaluation of the model performance: 

The accuracy score is 0.943227.

The Cohen's Kappa socre is 0.909061.

The micro precistion is 0.943227, the macro precision is 0.942859.

The micro recall is 0.943227, the macro recall is 0.931253.

The micro F1 score is 0.943227, the macro F1 score is 0.935312.

### 4 class, nsv: 1002
The confusion matrix is as follows: 

[[294   3   3  35]
 [  2 753  21  76]
 [ 11  44 502  35]
 [  2   7   3 142]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.95      0.88      0.91       335
          1       0.93      0.88      0.91       852
          2       0.95      0.85      0.90       592
          3       0.49      0.92      0.64       154

avg / total       0.91      0.87      0.88      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.874806.

The Cohen's Kappa socre is 0.819240.

The micro precistion is 0.874806, the macro precision is 0.831639.

The micro recall is 0.874806, the macro recall is 0.882866.

The micro F1 score is 0.874806, the macro F1 score is 0.839746.

### 5 class, nsv: 1488
The confusion matrix is as follows: 

[[248   5   2  12  68]
 [  2 747  14  33  56]
 [  3  11 468   2 108]
 [  7  19   2  96  30]
 [  4   5   0   2 109]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.74      0.83       335
          1       0.95      0.88      0.91       852
          2       0.96      0.79      0.87       592
          3       0.66      0.62      0.64       154
          4       0.29      0.91      0.44       120

avg / total       0.89      0.81      0.84      2053




Here is the evaluation of the model performance: 

The accuracy score is 0.812470.

The Cohen's Kappa socre is 0.745136.

The micro precistion is 0.812470, the macro precision is 0.761480.

The micro recall is 0.812470, the macro recall is 0.787862.

The micro F1 score is 0.812470, the macro F1 score is 0.738797.

### 6 class, nsv: 1746
The confusion matrix is as follows: 

[[274   1   4  12  36   8]
 [  2 472  12  27  14 325]
 [  3   5 512   2  31  39]
 [  3  16   1 107  18   9]
 [  8   2   2   4 101   3]
 [  1   1   0   0   1 164]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.82      0.88       335
          1       0.95      0.55      0.70       852
          2       0.96      0.86      0.91       592
          3       0.70      0.69      0.70       154
          4       0.50      0.84      0.63       120
          5       0.30      0.98      0.46       167

avg / total       0.86      0.73      0.76      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.734234.

The Cohen's Kappa socre is 0.668747.

The micro precistion is 0.734234, the macro precision is 0.726867.

The micro recall is 0.734234, the macro recall is 0.792546.

The micro F1 score is 0.734234, the macro F1 score is 0.712399.

## build the support data from all the previous data, lambda: 1, based on SVM
### 2 class
The confusion matrix is as follows: 

[[315  20]
 [  6 846]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.98      0.94      0.96       335
          1       0.98      0.99      0.98       852

avg / total       0.98      0.98      0.98      1187




Here is the evaluation of the model performance: 

The accuracy score is 0.978096.

The Cohen's Kappa socre is 0.945242.

The micro precistion is 0.978096, the macro precision is 0.979107.

The micro recall is 0.978096, the macro recall is 0.966628.

The micro F1 score is 0.978096, the macro F1 score is 0.972616.

### 3 class, nsv: 188
The confusion matrix is as follows: 

[[285  14  36]
 [ 10 810  32]
 [  3  15 574]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.96      0.85      0.90       335
          1       0.97      0.95      0.96       852
          2       0.89      0.97      0.93       592

avg / total       0.94      0.94      0.94      1779




Here is the evaluation of the model performance: 

The accuracy score is 0.938168.

The Cohen's Kappa socre is 0.900671.

The micro precistion is 0.938168, the macro precision is 0.938631.

The micro recall is 0.938168, the macro recall is 0.923682.

The micro F1 score is 0.938168, the macro F1 score is 0.929598.

### 4 class, nsv: 1032
The confusion matrix is as follows: 

[[300   3   1  31]
 [  3 754   5  90]
 [  5  22 492  73]
 [  2   5   2 145]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.97      0.90      0.93       335
          1       0.96      0.88      0.92       852
          2       0.98      0.83      0.90       592
          3       0.43      0.94      0.59       154

avg / total       0.93      0.87      0.89      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.874806.

The Cohen's Kappa socre is 0.821215.

The micro precistion is 0.874806, the macro precision is 0.835301.

The micro recall is 0.874806, the macro recall is 0.888285.

The micro F1 score is 0.874806, the macro F1 score is 0.835332.

### 5 class, nsv: 1266
The confusion matrix is as follows: 

[[281   3   6  20  25]
 [  4 725   9  28  86]
 [  2   9 502   4  75]
 [  7   9   3 113  22]
 [  6   1   9   5  99]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.84      0.89       335
          1       0.97      0.85      0.91       852
          2       0.95      0.85      0.90       592
          3       0.66      0.73      0.70       154
          4       0.32      0.82      0.46       120

avg / total       0.90      0.84      0.86      2053




Here is the evaluation of the model performance: 

The accuracy score is 0.837798.

The Cohen's Kappa socre is 0.779587.

The micro precistion is 0.837798, the macro precision is 0.768671.

The micro recall is 0.837798, the macro recall is 0.819297.

The micro F1 score is 0.837798, the macro F1 score is 0.769743.

### 6 class, nsv: 1944
The confusion matrix is as follows: 

[[269   5  12   7  34   8]
 [  4 680   3   8  31 126]
 [  1  23 363   1 188  16]
 [  2  17   2 115   4  14]
 [  5   2   1   3 104   5]
 [  2  14   0   0  10 141]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.95      0.80      0.87       335
          1       0.92      0.80      0.85       852
          2       0.95      0.61      0.75       592
          3       0.86      0.75      0.80       154
          4       0.28      0.87      0.42       120
          5       0.45      0.84      0.59       167

avg / total       0.86      0.75      0.78      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.753153.

The Cohen's Kappa socre is 0.684811.

The micro precistion is 0.753153, the macro precision is 0.735723.

The micro recall is 0.753153, the macro recall is 0.778669.

The micro F1 score is 0.753153, the macro F1 score is 0.713977.

## build the support data from all the previous data, lambda: 100000, based on SVM
### 2 class
The confusion matrix is as follows: 

[[311  24]
 [  5 847]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.98      0.93      0.96       335
          1       0.97      0.99      0.98       852

avg / total       0.98      0.98      0.98      1187




Here is the evaluation of the model performance: 

The accuracy score is 0.975569.

The Cohen's Kappa socre is 0.938642.

The micro precistion is 0.975569, the macro precision is 0.978311.

The micro recall is 0.975569, the macro recall is 0.961245.

The micro F1 score is 0.975569, the macro F1 score is 0.969311.

### 3 class, nsv: 198
The confusion matrix is as follows: 

[[297   4  34]
 [  5 793  54]
 [  1   4 587]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.98      0.89      0.93       335
          1       0.99      0.93      0.96       852
          2       0.87      0.99      0.93       592

avg / total       0.95      0.94      0.94      1779




Here is the evaluation of the model performance: 

The accuracy score is 0.942664.

The Cohen's Kappa socre is 0.908414.

The micro precistion is 0.942664, the macro precision is 0.946613.

The micro recall is 0.942664, the macro recall is 0.936291.

The micro F1 score is 0.942664, the macro F1 score is 0.939033.

### 4 class, nsv: 985
The confusion matrix is as follows: 

[[293  11   3  28]
 [  4 703   9 136]
 [  8  53 497  34]
 [  5   9   3 137]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.95      0.87      0.91       335
          1       0.91      0.83      0.86       852
          2       0.97      0.84      0.90       592
          3       0.41      0.89      0.56       154

avg / total       0.89      0.84      0.86      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.843249.

The Cohen's Kappa socre is 0.776177.

The micro precistion is 0.843249, the macro precision is 0.807687.

The micro recall is 0.843249, the macro recall is 0.857220.

The micro F1 score is 0.843249, the macro F1 score is 0.808213.

### 5 class, nsv: 1409
The confusion matrix is as follows: 

[[294   6   3  14  18]
 [  3 794   4  11  40]
 [  7  25 473   1  86]
 [ 13  23   2  95  21]
 [  4   5   1  12  98]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.92      0.88      0.90       335
          1       0.93      0.93      0.93       852
          2       0.98      0.80      0.88       592
          3       0.71      0.62      0.66       154
          4       0.37      0.82      0.51       120

avg / total       0.89      0.85      0.87      2053




Here is the evaluation of the model performance: 

The accuracy score is 0.854359.

The Cohen's Kappa socre is 0.798245.

The micro precistion is 0.854359, the macro precision is 0.782585.

The micro recall is 0.854359, the macro recall is 0.808415.

The micro F1 score is 0.854359, the macro F1 score is 0.776298.

### 6 class, nsv: 1717
The confusion matrix is as follows: 

[[278   3   8  20  24   2]
 [  3 638  19  16  11 165]
 [  0  12 520   4  31  25]
 [  5   7   5 119  13   5]
 [  4   3   3   8  95   7]
 [  1   1   1   0   2 162]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.96      0.83      0.89       335
          1       0.96      0.75      0.84       852
          2       0.94      0.88      0.91       592
          3       0.71      0.77      0.74       154
          4       0.54      0.79      0.64       120
          5       0.44      0.97      0.61       167

avg / total       0.87      0.82      0.83      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.816216.

The Cohen's Kappa socre is 0.763390.

The micro precistion is 0.816216, the macro precision is 0.757732.

The micro recall is 0.816216, the macro recall is 0.831918.

The micro F1 score is 0.816216, the macro F1 score is 0.771166.

## build the support data from all the previous data, lambda: 2000, based on SVM
### 2 class
The confusion matrix is as follows: 

[[317  18]
 [  5 847]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.98      0.95      0.96       335
          1       0.98      0.99      0.99       852

avg / total       0.98      0.98      0.98      1187




Here is the evaluation of the model performance: 

The accuracy score is 0.980623.

The Cohen's Kappa socre is 0.951604.

The micro precistion is 0.980623, the macro precision is 0.981831.

The micro recall is 0.980623, the macro recall is 0.970200.

The micro F1 score is 0.980623, the macro F1 score is 0.975798.

### 3 class
The confusion matrix is as follows: 

[[276   2  57]
 [  9 740 103]
 [  4   2 586]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.96      0.82      0.88       335
          1       0.99      0.87      0.93       852
          2       0.79      0.99      0.88       592

avg / total       0.92      0.90      0.90      1779




Here is the evaluation of the model performance: 

The accuracy score is 0.900506.

The Cohen's Kappa socre is 0.841966.

The micro precistion is 0.900506, the macro precision is 0.911721.

The micro recall is 0.900506, the macro recall is 0.894097.

The micro F1 score is 0.900506, the macro F1 score is 0.895956.

### 4 class
The confusion matrix is as follows: 

[[299   6   9  21]
 [  8 686   7 151]
 [ 23  32 486  51]
 [  3   3   2 146]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.90      0.89      0.90       335
          1       0.94      0.81      0.87       852
          2       0.96      0.82      0.89       592
          3       0.40      0.95      0.56       154

avg / total       0.90      0.84      0.85      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.836524.

The Cohen's Kappa socre is 0.769528.

The micro precistion is 0.836524, the macro precision is 0.800363.

The micro recall is 0.836524, the macro recall is 0.866675.

The micro F1 score is 0.836524, the macro F1 score is 0.802323.

### 5 class
The confusion matrix is as follows: 

[[232  21   2   4  76]
 [  6 766   6  46  28]
 [  3   9 401   6 173]
 [  3  15   0 120  16]
 [  3   4   1   3 109]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.69      0.80       335
          1       0.94      0.90      0.92       852
          2       0.98      0.68      0.80       592
          3       0.67      0.78      0.72       154
          4       0.27      0.91      0.42       120

avg / total       0.89      0.79      0.82      2053




Here is the evaluation of the model performance: 

The accuracy score is 0.792986.

The Cohen's Kappa socre is 0.720269.

The micro precistion is 0.792986, the macro precision is 0.759747.

The micro recall is 0.792986, the macro recall is 0.791303.

The micro F1 score is 0.792986, the macro F1 score is 0.731002.

### 6 class
The confusion matrix is as follows: 

[[288   4  14   8  10  11]
 [  4 666  14  36  17 115]
 [  2   7 546   4  21  12]
 [  4  12  11 106  15   6]
 [  6   2  13   4  92   3]
 [  1   0   1   0   1 164]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.86      0.90       335
          1       0.96      0.78      0.86       852
          2       0.91      0.92      0.92       592
          3       0.67      0.69      0.68       154
          4       0.59      0.77      0.67       120
          5       0.53      0.98      0.69       167

avg / total       0.87      0.84      0.85      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.838739.

The Cohen's Kappa socre is 0.790184.

The micro precistion is 0.838739, the macro precision is 0.767927.

The micro recall is 0.838739, the macro recall is 0.833451.

The micro F1 score is 0.838739, the macro F1 score is 0.785413.



## result of EWC only, lambda: 1000
### 2 class
The confusion matrix is as follows: 

[[324  11]
 [  9 843]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.97      0.97      0.97       335
          1       0.99      0.99      0.99       852

avg / total       0.98      0.98      0.98      1187




Here is the evaluation of the model performance: 

The accuracy score is 0.983151.

The Cohen's Kappa socre is 0.958337.

The micro precistion is 0.983151, the macro precision is 0.980046.

The micro recall is 0.983151, the macro recall is 0.978300.

The micro F1 score is 0.983151, the macro F1 score is 0.979168.


### 3 class
The confusion matrix is as follows: 

[[  0   0 335]
 [  0   0 852]
 [  0   0 592]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.00      0.00      0.00       335
          1       0.00      0.00      0.00       852
          2       0.33      1.00      0.50       592

avg / total       0.11      0.33      0.17      1779




Here is the evaluation of the model performance: 

The accuracy score is 0.332771.

The Cohen's Kappa socre is 0.000000.

The micro precistion is 0.332771, the macro precision is 0.110924.

The micro recall is 0.332771, the macro recall is 0.333333.

The micro F1 score is 0.332771, the macro F1 score is 0.166456.

### 4 class
The confusion matrix is as follows: 

[[  0   0   0 335]
 [  0   0   0 852]
 [  0   0   0 592]
 [  0   0   0 154]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.00      0.00      0.00       335
          1       0.00      0.00      0.00       852
          2       0.00      0.00      0.00       592
          3       0.08      1.00      0.15       154

avg / total       0.01      0.08      0.01      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.079669.

The Cohen's Kappa socre is 0.000000.

The micro precistion is 0.079669, the macro precision is 0.019917.

The micro recall is 0.079669, the macro recall is 0.250000.

The micro F1 score is 0.079669, the macro F1 score is 0.036895.

### 5 class
The confusion matrix is as follows: 

[[  0   0   0   0 335]
 [  0   0   0   0 852]
 [  0   0   0   0 592]
 [  0   0   0   0 154]
 [  0   0   0   0 120]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.00      0.00      0.00       335
          1       0.00      0.00      0.00       852
          2       0.00      0.00      0.00       592
          3       0.00      0.00      0.00       154
          4       0.06      1.00      0.11       120

avg / total       0.00      0.06      0.01      2053




Here is the evaluation of the model performance: 

The accuracy score is 0.058451.

The Cohen's Kappa socre is 0.000000.

The micro precistion is 0.058451, the macro precision is 0.011690.

The micro recall is 0.058451, the macro recall is 0.200000.

The micro F1 score is 0.058451, the macro F1 score is 0.022089.

### 6 class
The confusion matrix is as follows: 

[[  0   0   0   0   0 335]
 [  0   0   0   0   0 852]
 [  0   0   0   0   0 592]
 [  0   0   0   0   0 154]
 [  0   0   0   0   0 120]
 [  0   0   0   0   0 167]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.00      0.00      0.00       335
          1       0.00      0.00      0.00       852
          2       0.00      0.00      0.00       592
          3       0.00      0.00      0.00       154
          4       0.00      0.00      0.00       120
          5       0.08      1.00      0.14       167

avg / total       0.01      0.08      0.01      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.075225.

The Cohen's Kappa socre is 0.000000.

The micro precistion is 0.075225, the macro precision is 0.012538.

The micro recall is 0.075225, the macro recall is 0.166667.

The micro F1 score is 0.075225, the macro F1 score is 0.023321.


## icarl only, examplar size: 2000
### 2 class
The confusion matrix is as follows: 

[[330   5]
 [ 23 829]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.93      0.99      0.96       335
          1       0.99      0.97      0.98       852

avg / total       0.98      0.98      0.98      1187




Here is the evaluation of the model performance: 

The accuracy score is 0.976411.

The Cohen's Kappa socre is 0.942711.

The micro precistion is 0.976411, the macro precision is 0.964424.

The micro recall is 0.976411, the macro recall is 0.979040.

The micro F1 score is 0.976411, the macro F1 score is 0.971347.

### 3 class
The confusion matrix is as follows: 

[[210   1 124]
 [  4 603 245]
 [  2   4 586]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.97      0.63      0.76       335
          1       0.99      0.71      0.83       852
          2       0.61      0.99      0.76       592

avg / total       0.86      0.79      0.79      1779




Here is the evaluation of the model performance: 

The accuracy score is 0.786397.

The Cohen's Kappa socre is 0.663522.

The micro precistion is 0.786397, the macro precision is 0.859204.

The micro recall is 0.786397, the macro recall is 0.774826.

The micro F1 score is 0.786397, the macro F1 score is 0.781958.

### 4 class
The confusion matrix is as follows: 

[[206   1  50  78]
 [  2 554  43 253]
 [ 11  15 349 217]
 [  6   3   6 139]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.92      0.61      0.74       335
          1       0.97      0.65      0.78       852
          2       0.78      0.59      0.67       592
          3       0.20      0.90      0.33       154

avg / total       0.84      0.65      0.70      1933




Here is the evaluation of the model performance: 

The accuracy score is 0.645629.

The Cohen's Kappa socre is 0.527427.

The micro precistion is 0.645629, the macro precision is 0.715936.

The micro recall is 0.645629, the macro recall is 0.689321.

The micro F1 score is 0.645629, the macro F1 score is 0.628743.


### 5 class
The confusion matrix is as follows: 

[[195   1  38  65  36]
 [  5 488  78 246  35]
 [ 12  10 378 154  38]
 [  8  21  10  97  18]
 [  3   1   2   7 107]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.87      0.58      0.70       335
          1       0.94      0.57      0.71       852
          2       0.75      0.64      0.69       592
          3       0.17      0.63      0.27       154
          4       0.46      0.89      0.60       120

avg / total       0.79      0.62      0.66      2053




Here is the evaluation of the model performance: 

The accuracy score is 0.616171.

The Cohen's Kappa socre is 0.506923.

The micro precistion is 0.616171, the macro precision is 0.637175.

The micro recall is 0.616171, the macro recall is 0.662982.

The micro F1 score is 0.616171, the macro F1 score is 0.594230.

### 6 class
The confusion matrix is as follows: 

[[188   1  18  65  21  42]
 [  4 509  26 233   3  77]
 [  7   3 365 139   2  76]
 [  4  11  12 100   5  22]
 [  1   4   0   8  76  31]
 [  1   0   0   1   7 158]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.92      0.56      0.70       335
          1       0.96      0.60      0.74       852
          2       0.87      0.62      0.72       592
          3       0.18      0.65      0.29       154
          4       0.67      0.63      0.65       120
          5       0.39      0.95      0.55       167

avg / total       0.82      0.63      0.68      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.628829.

The Cohen's Kappa socre is 0.542116.

The micro precistion is 0.628829, the macro precision is 0.664509.

The micro recall is 0.628829, the macro recall is 0.667326.

The micro F1 score is 0.628829, the macro F1 score is 0.606897.

## random guess
The confusion matrix is as follows:

[[172 163]
 [427 425]]



The classification result for each class is as follows:

             precision    recall  f1-score   support

        0.0       0.29      0.51      0.37       335
        1.0       0.72      0.50      0.59       852

avg / total       0.60      0.50      0.53      1187




Here is the evaluation of the model performance:

The accuracy score is 0.502949.

The Cohen's Kappa score is 0.009894.

The micro precision is 0.502949, the macro precision is 0.504967.

The micro recall is 0.502949, the macro recall is 0.506130.

The micro F1 score is 0.502949, the macro F1 score is 0.479293.




The confusion matrix is as follows:

[[125 115  95]
 [276 291 285]
 [203 195 194]]



The classification result for each class is as follows:

             precision    recall  f1-score   support

        0.0       0.21      0.37      0.27       335
        1.0       0.48      0.34      0.40       852
        2.0       0.34      0.33      0.33       592

avg / total       0.38      0.34      0.35      1779




Here is the evaluation of the model performance:

The accuracy score is 0.342889.

The Cohen's Kappa score is 0.014683.

The micro precision is 0.342889, the macro precision is 0.343042.

The micro recall is 0.342889, the macro recall is 0.347462.

The micro F1 score is 0.342889, the macro F1 score is 0.333184.




The confusion matrix is as follows:

[[ 84  72  89  90]
 [222 202 210 218]
 [158 143 142 149]
 [ 30  44  39  41]]



The classification result for each class is as follows:

             precision    recall  f1-score   support

        0.0       0.17      0.25      0.20       335
        1.0       0.44      0.24      0.31       852
        2.0       0.30      0.24      0.26       592
        3.0       0.08      0.27      0.13       154

avg / total       0.32      0.24      0.26      1933




Here is the evaluation of the model performance:

The accuracy score is 0.242628.

The Cohen's Kappa score is -0.004450.

The micro precision is 0.242628, the macro precision is 0.246595.

The micro recall is 0.242628, the macro recall is 0.248484.

The micro F1 score is 0.242628, the macro F1 score is 0.225260.




The confusion matrix is as follows:

[[ 55  74  72  71  63]
 [205 153 153 183 158]
 [111 122 109 111 139]
 [ 34  28  38  31  23]
 [ 26  18  26  32  18]]



The classification result for each class is as follows:

             precision    recall  f1-score   support

        0.0       0.13      0.16      0.14       335
        1.0       0.39      0.18      0.25       852
        2.0       0.27      0.18      0.22       592
        3.0       0.07      0.20      0.11       154
        4.0       0.04      0.15      0.07       120

avg / total       0.27      0.18      0.20      2053




Here is the evaluation of the model performance:

The accuracy score is 0.178276.

The Cohen's Kappa score is -0.023395.

The micro precision is 0.178276, the macro precision is 0.181228.

The micro recall is 0.178276, the macro recall is 0.175835.

The micro F1 score is 0.178276, the macro F1 score is 0.156964.




The confusion matrix is as follows:

[[ 52  70  58  57  48  50]
 [133 155 144 141 124 155]
 [ 93  97  99 112  95  96]
 [ 24  25  28  20  28  29]
 [ 23  13  22  26  16  20]
 [ 33  24  29  30  26  25]]



The classification result for each class is as follows:

             precision    recall  f1-score   support

        0.0       0.15      0.16      0.15       335
        1.0       0.40      0.18      0.25       852
        2.0       0.26      0.17      0.20       592
        3.0       0.05      0.13      0.07       154
        4.0       0.05      0.13      0.07       120
        5.0       0.07      0.15      0.09       167

avg / total       0.26      0.17      0.19      2220




Here is the evaluation of the model performance:

The accuracy score is 0.165315.

The Cohen's Kappa score is -0.004843.

The micro precision is 0.165315, the macro precision is 0.162564.

The micro recall is 0.165315, the macro recall is 0.152880.

The micro F1 score is 0.165315, the macro F1 score is 0.140155.


## underfitting
### test on the final training data, icarl
The confusion matrix is as follows: 

[[ 326    0    0    0    0    0]
 [   0  820    0    0    0   10]
 [   0    0  577    0    0    0]
 [   0    0   12  134    0    3]
 [   0    0    0    0  116    0]
 [   1    0    5    0    0 1493]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       1.00      1.00      1.00       326
          1       1.00      0.99      0.99       830
          2       0.97      1.00      0.99       577
          3       1.00      0.90      0.95       149
          4       1.00      1.00      1.00       116
          5       0.99      1.00      0.99      1499

avg / total       0.99      0.99      0.99      3497




Here is the evaluation of the model performance: 

The accuracy score is 0.991135.

The Cohen's Kappa socre is 0.987692.

The micro precistion is 0.991135, the macro precision is 0.993282.

The micro recall is 0.991135, the macro recall is 0.980546.

The micro F1 score is 0.991135, the macro F1 score is 0.986427.

### test on test data, icarl
The confusion matrix is as follows: 

[[195   0  48  18   5  69]
 [  4 470 107  69   0 202]
 [  1   1 438  32   2 118]
 [  6   9  26  67   1  45]
 [  5   4   1   4  62  44]
 [  2   0   1   0   0 164]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.92      0.58      0.71       335
          1       0.97      0.55      0.70       852
          2       0.71      0.74      0.72       592
          3       0.35      0.44      0.39       154
          4       0.89      0.52      0.65       120
          5       0.26      0.98      0.41       167

avg / total       0.79      0.63      0.66      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.628829.

The Cohen's Kappa socre is 0.534791.

The micro precistion is 0.628829, the macro precision is 0.680946.

The micro recall is 0.628829, the macro recall is 0.634561.

The micro F1 score is 0.628829, the macro F1 score is 0.597509.



### test on all training data, icarl
The confusion matrix is as follows: 

[[1573    6  591  130   39  669]
 [  21 4195 1105  658    5 1681]
 [  13   39 4022  255   22  974]
 [  15   65  228  649   11  410]
 [  50    6    9   23  552  433]
 [   1    0    5    0    0 1493]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.52      0.67      3008
          1       0.97      0.55      0.70      7665
          2       0.67      0.76      0.71      5325
          3       0.38      0.47      0.42      1378
          4       0.88      0.51      0.65      1073
          5       0.26      1.00      0.42      1499

avg / total       0.79      0.63      0.66     19948




Here is the evaluation of the model performance: 

The accuracy score is 0.625827.

The Cohen's Kappa socre is 0.529697.

The micro precistion is 0.625827, the macro precision is 0.684657.

The micro recall is 0.625827, the macro recall is 0.634492.

The micro F1 score is 0.625827, the macro F1 score is 0.595142.

### test on the final training set, svm
The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.99      0.99      0.99       250
          1       0.99      0.99      0.99       518
          2       0.94      1.00      0.97       458
          3       0.99      0.98      0.99       343
          4       0.99      0.95      0.97       430
          5       1.00      0.99      0.99      1499

avg / total       0.99      0.99      0.99      3498




Here is the evaluation of the model performance: 

The accuracy score is 0.986564.

The Cohen's Kappa socre is 0.982040.

The micro precistion is 0.986564, the macro precision is 0.984266.

The micro recall is 0.986564, the macro recall is 0.984700.

The micro F1 score is 0.986564, the macro F1 score is 0.984246.

### test on the test data set, svm
The confusion matrix is as follows: 

[[288   4  14   8  10  11]
 [  4 666  14  36  17 115]
 [  2   7 546   4  21  12]
 [  4  12  11 106  15   6]
 [  6   2  13   4  92   3]
 [  1   0   1   0   1 164]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.86      0.90       335
          1       0.96      0.78      0.86       852
          2       0.91      0.92      0.92       592
          3       0.67      0.69      0.68       154
          4       0.59      0.77      0.67       120
          5       0.53      0.98      0.69       167

avg / total       0.87      0.84      0.85      2220




Here is the evaluation of the model performance: 

The accuracy score is 0.838739.

The Cohen's Kappa socre is 0.790184.

The micro precistion is 0.838739, the macro precision is 0.767927.

The micro recall is 0.838739, the macro recall is 0.833451.

The micro F1 score is 0.838739, the macro F1 score is 0.785413.



### test on the all the training set, svm
The confusion matrix is as follows: 

[[2836   27   29   21   45   50]
 [  15 6607   24  141   20  858]
 [   3   29 5182    4   20   87]
 [   1   26   25 1245   36   45]
 [   7    3   50    5  997   11]
 [   2    1   11    0    1 1484]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.99      0.94      0.97      3008
          1       0.99      0.86      0.92      7665
          2       0.97      0.97      0.97      5325
          3       0.88      0.90      0.89      1378
          4       0.89      0.93      0.91      1073
          5       0.59      0.99      0.74      1499

avg / total       0.94      0.92      0.92     19948




Here is the evaluation of the model performance: 

The accuracy score is 0.919942.

The Cohen's Kappa socre is 0.894764.

The micro precistion is 0.919942, the macro precision is 0.884478.

The micro recall is 0.919942, the macro recall is 0.933430.

The micro F1 score is 0.919942, the macro F1 score is 0.899398.



# Only support data

The confusion matrix is as follows:

[[331   4]
 [ 20 832]]



The classification result for each class is as follows:

             precision    recall  f1-score   support

          0       0.94      0.99      0.97       335
          1       1.00      0.98      0.99       852

avg / total       0.98      0.98      0.98      1187




Here is the evaluation of the model performance:

The accuracy score is 0.979781.

The Cohen's Kappa socre is 0.950807.

The micro precistion is 0.979781, the macro precision is 0.969118.

The micro recall is 0.979781, the macro recall is 0.982293.

The micro F1 score is 0.979781, the macro F1 score is 0.975398.

The confusion matrix is as follows:

[[328   2   5]
 [ 37 805  10]
 [ 41   8 543]]



The classification result for each class is as follows:

             precision    recall  f1-score   support

          0       0.81      0.98      0.89       335
          1       0.99      0.94      0.97       852
          2       0.97      0.92      0.94       592

avg / total       0.95      0.94      0.94      1779




Here is the evaluation of the model performance:

The accuracy score is 0.942102.

The Cohen's Kappa socre is 0.908570.

The micro precistion is 0.942102, the macro precision is 0.922910.

The micro recall is 0.942102, the macro recall is 0.947057.

The micro F1 score is 0.942102, the macro F1 score is 0.931815.


The confusion matrix is as follows:

[[293   3  19  20]
 [  3 807  17  25]
 [  8  40 534  10]
 [  5  10  12 127]]



The classification result for each class is as follows:

             precision    recall  f1-score   support

          0       0.95      0.87      0.91       335
          1       0.94      0.95      0.94       852
          2       0.92      0.90      0.91       592
          3       0.70      0.82      0.76       154

avg / total       0.91      0.91      0.91      1933




Here is the evaluation of the model performance:

The accuracy score is 0.911019.

The Cohen's Kappa socre is 0.868466.

The micro precistion is 0.911019, the macro precision is 0.875480.

The micro recall is 0.911019, the macro recall is 0.887128.

The micro F1 score is 0.911019, the macro F1 score is 0.879589.


The confusion matrix is as follows:

[[243  17   4  19  52]
 [  4 755   4  45  44]
 [  6  47 522  10   7]
 [ 11  18   4 111  10]
 [  3   5   0   7 105]]



The classification result for each class is as follows:

             precision    recall  f1-score   support

          0       0.91      0.73      0.81       335
          1       0.90      0.89      0.89       852
          2       0.98      0.88      0.93       592
          3       0.58      0.72      0.64       154
          4       0.48      0.88      0.62       120

avg / total       0.87      0.85      0.85      2053




Here is the evaluation of the model performance:

The accuracy score is 0.845592.

The Cohen's Kappa socre is 0.785647.

The micro precistion is 0.845592, the macro precision is 0.768818.

The micro recall is 0.845592, the macro recall is 0.817812.

The micro F1 score is 0.845592, the macro F1 score is 0.777757.

The confusion matrix is as follows:

[[293   8  15   8  11   0]
 [  2 456  22  38   5 329]
 [  3  16 539  12   4  18]
 [  2   8   6 128   9   1]
 [  3   8   4   7  97   1]
 [  2   1   1   0   1 162]]



The classification result for each class is as follows:

             precision    recall  f1-score   support

          0       0.96      0.87      0.92       335
          1       0.92      0.54      0.68       852
          2       0.92      0.91      0.91       592
          3       0.66      0.83      0.74       154
          4       0.76      0.81      0.79       120
          5       0.32      0.97      0.48       167

avg / total       0.85      0.75      0.77      2220




Here is the evaluation of the model performance:

The accuracy score is 0.754505.

The Cohen's Kappa socre is 0.691743.

The micro precistion is 0.754505, the macro precision is 0.756734.

The micro recall is 0.754505, the macro recall is 0.821646.

The micro F1 score is 0.754505, the macro F1 score is 0.751178.