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

## result of EWC only
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


