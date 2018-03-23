## original all data, normal method
The confusion matrix is as follows: 

[[65  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 1 25  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  1 23  0  0  1  0  1  0  0  0  0  0  0  0  0  0  1  0  0  0]
 [ 1  0  0 12  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  1  2  0 10  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 1  0  0  0  0 40  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0]
 [ 0  1  0  0  0  1  5  1  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  1  1 13  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  2  0  0  6  1  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  5  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  1  0  0  0  1 11  0  0  1  0  0  0  0  0  0  0]
 [ 0  1  0  0  0  3  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0]
 [ 1  0  0  0  0  0  0  0  0  0  2  0  9  2  0  0  0  0  0  0  0]
 [ 0  0  2  0  0  1  1  0  0  0  0  0  0 39  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  5  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  5  0  0  0  0  0]
 [ 2  1  0  0  0  0  0  0  0  0  0  0  0  1  0  0 11  0  0  0  0]
 [ 0  0  1  1  0  0  0  1  0  0  0  0  0  0  0  0  0  4  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0]
 [ 0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0]
 [ 0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  1]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.88      0.98      0.93        66
          1       0.83      0.93      0.88        27
          2       0.79      0.77      0.78        30
          3       0.86      0.86      0.86        14
          4       1.00      0.67      0.80        15
          5       0.75      0.95      0.84        42
          6       0.56      0.62      0.59         8
          7       0.76      0.81      0.79        16
          8       1.00      0.67      0.80         9
          9       0.71      1.00      0.83         5
         10       0.85      0.79      0.81        14
         11       0.50      0.20      0.29         5
         12       1.00      0.64      0.78        14
         13       0.89      0.91      0.90        43
         14       1.00      1.00      1.00         5
         15       1.00      1.00      1.00         5
         16       1.00      0.73      0.85        15
         17       0.80      0.57      0.67         7
         18       0.00      0.00      0.00         1
         19       1.00      0.50      0.67         2
         20       1.00      0.50      0.67         2

avg / total       0.85      0.84      0.84       345




Here is the evaluation of the model performance: 

The accuracy score is 0.843478.

The Cohen's Kappa socre is 0.825875.

The micro precistion is 0.843478, the macro precision is 0.818273.

The micro recall is 0.843478, the macro recall is 0.718958.

The micro F1 score is 0.843478, the macro F1 score is 0.748570.


## 9 classes with KNN. 
The confusion matrix is as follows: 

[[60  0  3  0  1  2  0  0  0]
 [ 2 23  0  0  1  0  1  0  0]
 [ 3  0 24  0  1  1  0  1  0]
 [ 0  0  2 10  1  0  0  1  0]
 [ 0  1  1  1 11  1  0  0  0]
 [ 0  0  0  0  2 40  0  0  0]
 [ 0  0  0  0  1  0  6  1  0]
 [ 0  0  0  0  1  0  1 14  0]
 [ 0  0  0  1  0  1  0  0  7]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.92      0.91      0.92        66
          1       0.96      0.85      0.90        27
          2       0.80      0.80      0.80        30
          3       0.83      0.71      0.77        14
          4       0.58      0.73      0.65        15
          5       0.89      0.95      0.92        42
          6       0.75      0.75      0.75         8
          7       0.82      0.88      0.85        16
          8       1.00      0.78      0.88         9

avg / total       0.87      0.86      0.86       227




Here is the evaluation of the model performance: 

The accuracy score is 0.859031.

The Cohen's Kappa socre is 0.830869.

The micro precistion is 0.859031, the macro precision is 0.839568.

The micro recall is 0.859031, the macro recall is 0.818191.

The micro F1 score is 0.859031, the macro F1 score is 0.825256.


## Performance of incrementally using all data to train
### 5class
The confusion matrix is as follows: 

[[62  0  2  0  2]
 [ 2 22  2  0  1]
 [ 6  0 22  0  2]
 [ 0  0  7  7  0]
 [ 1  1  2  0 11]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.87      0.94      0.91        66
          1       0.96      0.81      0.88        27
          2       0.63      0.73      0.68        30
          3       1.00      0.50      0.67        14
          4       0.69      0.73      0.71        15

avg / total       0.83      0.82      0.81       152




Here is the evaluation of the model performance: 

The accuracy score is 0.815789.

The Cohen's Kappa socre is 0.740630.

The micro precistion is 0.815789, the macro precision is 0.829167.

The micro recall is 0.815789, the macro recall is 0.744175.

The micro F1 score is 0.815789, the macro F1 score is 0.767675.

### 10 class
The confusion matrix is as follows: 

[[60  0  2  0  0  3  0  0  1  0]
 [ 2 20  0  0  0  1  2  0  1  1]
 [ 4  0 22  0  1  1  1  1  0  0]
 [ 0  0  4  7  1  0  0  2  0  0]
 [ 0  1  0  0 11  1  1  1  0  0]
 [ 0  1  0  0  2 38  0  0  1  0]
 [ 0  0  0  0  0  4  2  0  0  2]
 [ 2  0  1  0  1  1  0 10  1  0]
 [ 0  0  0  0  0  2  4  0  2  1]
 [ 0  0  0  0  0  0  0  0  0  5]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.88      0.91      0.90        66
          1       0.91      0.74      0.82        27
          2       0.76      0.73      0.75        30
          3       1.00      0.50      0.67        14
          4       0.69      0.73      0.71        15
          5       0.75      0.90      0.82        42
          6       0.20      0.25      0.22         8
          7       0.71      0.62      0.67        16
          8       0.33      0.22      0.27         9
          9       0.56      1.00      0.71         5

avg / total       0.78      0.76      0.76       232




Here is the evaluation of the model performance: 

The accuracy score is 0.762931.

The Cohen's Kappa socre is 0.716375.

The micro precistion is 0.762931, the macro precision is 0.678584.

The micro recall is 0.762931, the macro recall is 0.661848.

The micro F1 score is 0.762931, the macro F1 score is 0.652100.

### 15 class
The confusion matrix is as follows: 

[[63  0  0  0  0  1  0  0  0  0  0  0  1  1  0]
 [ 1 23  1  0  0  0  1  0  0  0  0  0  1  0  0]
 [ 4  0 20  0  0  1  0  2  0  0  0  0  0  3  0]
 [ 0  0  3  7  0  0  0  0  0  0  0  0  0  4  0]
 [ 0  1  0  0  9  2  0  0  0  0  0  1  0  2  0]
 [ 1  1  0  0  0 34  0  0  0  0  1  3  0  2  0]
 [ 0  1  1  0  0  1  1  0  0  0  1  0  0  2  1]
 [ 0  0  0  0  1  2  0 11  0  0  1  0  0  1  0]
 [ 0  0  0  0  0  0  0  0  3  1  0  0  5  0  0]
 [ 0  1  0  0  0  0  0  0  0  4  0  0  0  0  0]
 [ 0  1  0  0  0  0  0  0  0  0  7  0  1  4  1]
 [ 0  1  0  0  0  4  0  0  0  0  0  0  0  0  0]
 [ 1  0  0  0  0  0  0  0  1  0  2  0  8  2  0]
 [ 1  0  0  0  0  0  0  0  0  0  0  0  0 42  0]
 [ 1  0  0  0  0  0  0  0  0  0  0  0  1  0  3]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.88      0.95      0.91        66
          1       0.79      0.85      0.82        27
          2       0.80      0.67      0.73        30
          3       1.00      0.50      0.67        14
          4       0.90      0.60      0.72        15
          5       0.76      0.81      0.78        42
          6       0.50      0.12      0.20         8
          7       0.85      0.69      0.76        16
          8       0.75      0.33      0.46         9
          9       0.80      0.80      0.80         5
         10       0.58      0.50      0.54        14
         11       0.00      0.00      0.00         5
         12       0.47      0.57      0.52        14
         13       0.67      0.98      0.79        43
         14       0.60      0.60      0.60         5

avg / total       0.76      0.75      0.74       313




Here is the evaluation of the model performance: 

The accuracy score is 0.750799.

The Cohen's Kappa socre is 0.716449.

The micro precistion is 0.750799, the macro precision is 0.689360.

The micro recall is 0.750799, the macro recall is 0.598440.

The micro F1 score is 0.750799, the macro F1 score is 0.619815.

### 21 class (around 10% performance degradation compared to the all data, even through final all the data is visible, overfitting problem is in the convnets?)
The confusion matrix is as follows: 

[[63  0  0  0  0  2  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0]
 [ 1 21  1  0  0  1  1  0  0  0  0  0  1  0  0  0  1  0  0  0  0]
 [ 4  0 19  0  0  1  0  2  0  0  0  0  0  3  0  0  0  1  0  0  0]
 [ 0  0  2  6  1  0  0  0  0  0  0  0  0  5  0  0  0  0  0  0  0]
 [ 0  1  0  0 11  3  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 1  1  0  0  0 32  0  0  0  0  0  1  0  1  0  1  4  0  0  1  0]
 [ 0  0  0  0  0  1  1  0  0  1  0  1  0  1  0  0  3  0  0  0  0]
 [ 0  0  0  0  0  1  0 11  0  0  0  0  0  2  0  0  0  2  0  0  0]
 [ 0  0  0  0  0  0  0  0  5  1  0  0  0  0  0  0  0  0  3  0  0]
 [ 0  0  0  1  0  0  0  0  0  3  0  0  0  0  0  0  1  0  0  0  0]
 [ 0  0  0  1  0  1  0  0  0  0  8  1  0  0  0  1  2  0  0  0  0]
 [ 0  1  0  0  0  3  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0]
 [ 1  0  1  0  0  0  0  0  2  0  0  0  6  3  0  0  1  0  0  0  0]
 [ 1  0  0  0  2  0  0  0  0  0  0  0  0 35  0  0  2  1  0  2  0]
 [ 1  0  0  0  0  0  0  0  0  0  0  0  0  0  4  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  1  0  0  0  1  2  1  0  0  0  0]
 [ 1  0  0  1  0  0  0  0  0  0  0  0  0  2  0  0 11  0  0  0  0]
 [ 0  0  1  0  0  0  0  1  0  0  0  0  0  2  0  0  0  3  0  0  0]
 [ 0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  1  0  0  0]
 [ 0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  1  0  0  0  0]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.86      0.95      0.91        66
          1       0.88      0.78      0.82        27
          2       0.79      0.63      0.70        30
          3       0.67      0.43      0.52        14
          4       0.79      0.73      0.76        15
          5       0.71      0.76      0.74        42
          6       0.33      0.12      0.18         8
          7       0.73      0.69      0.71        16
          8       0.71      0.56      0.63         9
          9       0.60      0.60      0.60         5
         10       0.80      0.57      0.67        14
         11       0.00      0.00      0.00         5
         12       0.75      0.43      0.55        14
         13       0.65      0.81      0.72        43
         14       0.80      0.80      0.80         5
         15       0.50      0.40      0.44         5
         16       0.39      0.73      0.51        15
         17       0.38      0.43      0.40         7
         18       0.00      0.00      0.00         1
         19       0.00      0.00      0.00         2
         20       0.00      0.00      0.00         2

avg / total       0.71      0.70      0.69       345




Here is the evaluation of the model performance: 

The accuracy score is 0.698551.

The Cohen's Kappa socre is 0.665117.

The micro precistion is 0.698551, the macro precision is 0.540006.

The micro recall is 0.698551, the macro recall is 0.496828.

The micro F1 score is 0.698551, the macro F1 score is 0.507430.

### 21 class, reinitilize the last layer and with all the data available
The confusion matrix is as follows: 

[[62  0  0  0  0  2  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0]
 [ 1 21  1  0  0  1  1  0  0  0  0  0  1  0  0  0  1  0  0  0  0]
 [ 4  0 20  0  0  1  0  2  0  0  0  0  0  2  0  0  0  1  0  0  0]
 [ 0  0  5  4  0  0  0  0  0  0  0  0  0  4  0  0  1  0  0  0  0]
 [ 1  1  0  0  9  2  0  0  0  0  0  0  0  1  0  0  1  0  0  0  0]
 [ 1  0  0  0  0 36  0  0  0  0  0  0  0  1  0  1  2  0  0  1  0]
 [ 0  0  1  0  0  1  2  0  0  0  0  0  1  2  0  1  0  0  0  0  0]
 [ 0  0  0  0  0  3  0 12  0  0  0  0  0  0  0  0  0  1  0  0  0]
 [ 0  0  0  0  0  0  0  0  5  1  0  0  3  0  0  0  0  0  0  0  0]
 [ 0  0  0  1  0  0  0  0  0  3  0  0  0  0  0  0  1  0  0  0  0]
 [ 0  1  0  0  0  0  1  0  0  0  9  0  0  2  0  0  1  0  0  0  0]
 [ 0  1  0  0  0  3  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0]
 [ 2  0  1  0  0  0  0  0  0  0  0  0  8  2  0  0  1  0  0  0  0]
 [ 2  0  3  0  0  2  0  3  0  0  0  0  0 32  0  0  0  0  0  1  0]
 [ 1  0  0  0  0  0  0  0  0  0  0  0  0  0  4  0  0  0  0  0  0]
 [ 0  0  0  0  0  2  0  0  0  0  1  0  0  0  0  2  0  0  0  0  0]
 [ 2  0  1  1  0  1  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0]
 [ 1  0  2  0  0  0  0  1  0  0  0  0  0  1  0  0  0  2  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  1  0  0  0]
 [ 0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  1  0  0  0  0]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.81      0.94      0.87        66
          1       0.88      0.78      0.82        27
          2       0.59      0.67      0.62        30
          3       0.67      0.29      0.40        14
          4       1.00      0.60      0.75        15
          5       0.67      0.86      0.75        42
          6       0.40      0.25      0.31         8
          7       0.67      0.75      0.71        16
          8       1.00      0.56      0.71         9
          9       0.60      0.60      0.60         5
         10       0.82      0.64      0.72        14
         11       0.00      0.00      0.00         5
         12       0.62      0.57      0.59        14
         13       0.64      0.74      0.69        43
         14       1.00      0.80      0.89         5
         15       0.50      0.40      0.44         5
         16       0.53      0.67      0.59        15
         17       0.40      0.29      0.33         7
         18       0.00      0.00      0.00         1
         19       0.00      0.00      0.00         2
         20       0.00      0.00      0.00         2

avg / total       0.69      0.70      0.68       345




Here is the evaluation of the model performance: 

The accuracy score is 0.698551.

The Cohen's Kappa socre is 0.663225.

The micro precistion is 0.698551, the macro precision is 0.560396.

The micro recall is 0.698551, the macro recall is 0.494910.

The micro F1 score is 0.698551, the macro F1 score is 0.514247.


## incremental training with support data and no regularizer and no weight
### 5 class
The confusion matrix is as follows: 

[[62  0  1  0  3]
 [ 2 23  2  0  0]
 [ 3  1 25  0  1]
 [ 0  0  6  7  1]
 [ 0  1  4  1  9]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.93      0.94      0.93        66
          1       0.92      0.85      0.88        27
          2       0.66      0.83      0.74        30
          3       0.88      0.50      0.64        14
          4       0.64      0.60      0.62        15

avg / total       0.84      0.83      0.83       152




Here is the evaluation of the model performance: 

The accuracy score is 0.828947.

The Cohen's Kappa socre is 0.761136.

The micro precistion is 0.828947, the macro precision is 0.804225.

The micro recall is 0.828947, the macro recall is 0.744916.

The micro F1 score is 0.828947, the macro F1 score is 0.761859.

### 10 class
The confusion matrix is as follows: 

[[58  0  0  0  0  5  1  0  2  0]
 [ 1 17  0  0  0  3  3  0  2  1]
 [ 3  0 19  0  0  5  0  3  0  0]
 [ 0  0  2  4  0  2  1  4  0  1]
 [ 0  0  0  0  5  3  5  1  0  1]
 [ 0  0  0  0  0 42  0  0  0  0]
 [ 0  0  0  0  0  2  4  0  0  2]
 [ 0  0  0  0  0  3  0 12  0  1]
 [ 0  0  0  0  0  1  0  0  7  1]
 [ 0  0  0  0  0  0  1  0  0  4]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.94      0.88      0.91        66
          1       1.00      0.63      0.77        27
          2       0.90      0.63      0.75        30
          3       1.00      0.29      0.44        14
          4       1.00      0.33      0.50        15
          5       0.64      1.00      0.78        42
          6       0.27      0.50      0.35         8
          7       0.60      0.75      0.67        16
          8       0.64      0.78      0.70         9
          9       0.36      0.80      0.50         5

avg / total       0.82      0.74      0.74       232




Here is the evaluation of the model performance: 

The accuracy score is 0.741379.

The Cohen's Kappa socre is 0.691667.

The micro precistion is 0.741379, the macro precision is 0.734328.

The micro recall is 0.741379, the macro recall is 0.658858.

The micro F1 score is 0.741379, the macro F1 score is 0.636079.

### 15 class
The confusion matrix is as follows: 

[[55  0  0  0  0  2  0  3  6  0]
 [ 0 19  0  0  0  0  0  1  7  0]
 [ 3  0 21  0  0  3  0  1  2  0]
 [ 0  0  2  9  1  0  0  0  2  0]
 [ 0  1  0  0 10  1  0  2  1  0]
 [ 0  0  0  0  0  7  0  1  5  1]
 [ 0  0  0  0  0  0  3  1  1  0]
 [ 1  0  1  2  0  1  0  6  3  0]
 [ 1  0  0  0  1  0  0  2 39  0]
 [ 0  0  0  0  0  0  0  0  1  4]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.92      0.83      0.87        66
          1       0.95      0.70      0.81        27
          2       0.88      0.70      0.78        30
          3       0.82      0.64      0.72        14
          4       0.83      0.67      0.74        15
         10       0.50      0.50      0.50        14
         11       1.00      0.60      0.75         5
         12       0.35      0.43      0.39        14
         13       0.58      0.91      0.71        43
         14       0.80      0.80      0.80         5

avg / total       0.78      0.74      0.75       233




Here is the evaluation of the model performance: 

The accuracy score is 0.742489.

The Cohen's Kappa socre is 0.691935.

The micro precistion is 0.742489, the macro precision is 0.762821.

The micro recall is 0.742489, the macro recall is 0.678211.

The micro F1 score is 0.742489, the macro F1 score is 0.706623.

### 21 class
The confusion matrix is as follows: 

[[51  0  0  0  0  0  0  3  3  0  3  2  1  0  3  0]
 [ 0 14  0  0  0  0  0  0  4  0  0  9  0  0  0  0]
 [ 2  0 13  0  0  0  0  4  1  0  1  2  6  0  1  0]
 [ 0  0  2  3  0  0  0  0  0  0  0  5  4  0  0  0]
 [ 0  0  0  0  8  0  0  0  0  0  1  5  1  0  0  0]
 [ 1  0  0  0  0  6  0  0  2  0  2  3  0  0  0  0]
 [ 0  0  0  0  0  0  2  1  0  0  0  2  0  0  0  0]
 [ 1  0  0  0  0  1  0  5  2  0  1  3  1  0  0  0]
 [ 1  0  0  1  1  0  0  1 32  1  1  3  1  1  0  0]
 [ 0  0  0  0  0  0  0  0  0  3  0  1  0  1  0  0]
 [ 0  0  0  0  0  1  0  0  0  1  2  1  0  0  0  0]
 [ 1  0  0  0  0  1  0  0  2  0  0 10  1  0  0  0]
 [ 0  0  0  0  0  0  0  0  1  0  0  1  5  0  0  0]
 [ 0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  1  0  0  0  0  0  0  1  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.89      0.77      0.83        66
          1       1.00      0.52      0.68        27
          2       0.87      0.43      0.58        30
          3       0.75      0.21      0.33        14
          4       0.80      0.53      0.64        15
         10       0.60      0.43      0.50        14
         11       1.00      0.40      0.57         5
         12       0.36      0.36      0.36        14
         13       0.68      0.74      0.71        43
         14       0.60      0.60      0.60         5
         15       0.18      0.40      0.25         5
         16       0.20      0.67      0.31        15
         17       0.24      0.71      0.36         7
         18       0.00      0.00      0.00         1
         19       0.00      0.00      0.00         2
         20       0.00      0.00      0.00         2

avg / total       0.72      0.58      0.61       265




Here is the evaluation of the model performance: 

The accuracy score is 0.581132.

The Cohen's Kappa socre is 0.526252.

The micro precistion is 0.581132, the macro precision is 0.510837.

The micro recall is 0.581132, the macro recall is 0.423941.

The micro F1 score is 0.581132, the macro F1 score is 0.420164.


## support data with ewc, highly overfitting, low regularization
### 5 class
The confusion matrix is as follows: 

[[61  0  1  0  4]
 [ 2 22  0  0  3]
 [ 6  0 21  0  3]
 [ 0  0  6  7  1]
 [ 1  1  0  0 13]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.87      0.92      0.90        66
          1       0.96      0.81      0.88        27
          2       0.75      0.70      0.72        30
          3       1.00      0.50      0.67        14
          4       0.54      0.87      0.67        15

avg / total       0.84      0.82      0.82       152




Here is the evaluation of the model performance: 

The accuracy score is 0.815789.

The Cohen's Kappa socre is 0.743073.

The micro precistion is 0.815789, the macro precision is 0.823923.

The micro recall is 0.815789, the macro recall is 0.761145.

The micro F1 score is 0.815789, the macro F1 score is 0.766906.

### 10 class
The confusion matrix is as follows: 

[[54  0  3  0  0  2  2  3  0  2]
 [ 2 18  0  0  0  4  3  0  0  0]
 [ 1  0 22  0  0  3  0  4  0  0]
 [ 0  0  4  6  0  1  0  3  0  0]
 [ 0  0  0  0  8  3  2  2  0  0]
 [ 0  1  0  0  0 37  1  1  0  2]
 [ 0  1  0  0  1  2  3  0  0  1]
 [ 0  0  1  2  1  3  1  8  0  0]
 [ 0  0  0  0  0  0  0  1  7  1]
 [ 0  0  1  0  0  2  0  0  0  2]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.95      0.82      0.88        66
          1       0.90      0.67      0.77        27
          2       0.71      0.73      0.72        30
          3       0.75      0.43      0.55        14
          4       0.80      0.53      0.64        15
          5       0.65      0.88      0.75        42
          6       0.25      0.38      0.30         8
          7       0.36      0.50      0.42        16
          8       1.00      0.78      0.88         9
          9       0.25      0.40      0.31         5

avg / total       0.76      0.71      0.72       232




Here is the evaluation of the model performance: 

The accuracy score is 0.711207.

The Cohen's Kappa socre is 0.657508.

The micro precistion is 0.711207, the macro precision is 0.661981.

The micro recall is 0.711207, the macro recall is 0.611382.

The micro F1 score is 0.711207, the macro F1 score is 0.620199.

### 15 class
The confusion matrix is as follows: 

[[38  0  0  0  0  0  1  1  0  0  5  0  1 20  0]
 [ 1 17  0  0  0  0  1  0  0  0  0  0  7  1  0]
 [ 2  0  8  0  0  0  0  3  0  0 10  0  2  5  0]
 [ 0  0  4  0  0  0  0  2  0  0  2  0  3  2  1]
 [ 0  0  0  0  4  0  1  1  0  0  0  1  3  4  1]
 [ 0  0  0  0  0 27  0  0  0  0  1  4  5  4  1]
 [ 0  0  0  0  0  0  1  0  0  0  1  1  3  1  1]
 [ 0  0  0  0  0  0  0  7  0  0  5  1  2  1  0]
 [ 0  0  0  0  0  0  0  0  1  0  1  1  2  4  0]
 [ 0  0  1  0  0  0  0  0  0  1  0  0  1  1  1]
 [ 0  0  1  0  0  0  0  1  0  0  8  0  1  3  0]
 [ 0  0  0  0  0  1  0  0  0  0  0  3  1  0  0]
 [ 0  0  0  0  1  0  1  0  0  1  2  0  7  2  0]
 [ 0  0  0  0  0  0  1  2  0  0  3  0  2 35  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  4]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.93      0.58      0.71        66
          1       1.00      0.63      0.77        27
          2       0.57      0.27      0.36        30
          3       0.00      0.00      0.00        14
          4       0.80      0.27      0.40        15
          5       0.96      0.64      0.77        42
          6       0.17      0.12      0.14         8
          7       0.41      0.44      0.42        16
          8       1.00      0.11      0.20         9
          9       0.50      0.20      0.29         5
         10       0.21      0.57      0.31        14
         11       0.27      0.60      0.37         5
         12       0.17      0.50      0.26        14
         13       0.42      0.81      0.55        43
         14       0.44      0.80      0.57         5

avg / total       0.65      0.51      0.53       313




Here is the evaluation of the model performance: 

The accuracy score is 0.514377.

The Cohen's Kappa socre is 0.459302.

The micro precistion is 0.514377, the macro precision is 0.524023.

The micro recall is 0.514377, the macro recall is 0.436038.

The micro F1 score is 0.514377, the macro F1 score is 0.409030.

### 21 class
The confusion matrix is as follows: 

[[34  0  0  0  0  0  0  1  0  0  8  0  0 16  0  0  5  1  1  0  0]
 [ 1 12  0  0  0  0  0  0  0  0  0  2  1  1  0  0  9  0  0  1  0]
 [ 2  0  7  0  0  0  0  1  0  0  3  0  1  6  0  3  4  3  0  0  0]
 [ 0  0  0  0  0  0  0  2  0  0  0  0  0  2  1  0  5  4  0  0  0]
 [ 0  0  0  0  4  0  0  1  0  0  0  2  3  2  1  0  2  0  0  0  0]
 [ 0  0  0  0  0 24  0  0  0  0  0  4  4  3  0  1  4  0  0  1  1]
 [ 0  0  0  0  0  0  0  0  0  0  0  1  0  2  0  1  2  1  0  1  0]
 [ 0  0  0  0  0  0  0  5  0  0  2  1  1  1  0  1  1  2  1  1  0]
 [ 0  0  0  0  0  0  0  0  2  0  1  0  2  0  0  1  0  2  1  0  0]
 [ 0  0  0  0  0  0  0  0  0  1  0  0  1  1  0  2  0  0  0  0  0]
 [ 0  0  1  0  0  0  0  1  0  0  7  0  2  2  0  0  1  0  0  0  0]
 [ 0  0  0  0  0  1  0  0  0  0  0  3  1  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  1  0  1  0  0  0  0  0  6  3  0  0  2  0  0  1  0]
 [ 2  0  0  0  0  0  2  0  0  0  2  0  0 34  0  0  1  0  1  1  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  4  0  0  1  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  5  0  0  0  0  0]
 [ 1  0  0  0  0  0  1  0  0  0  0  0  3  2  0  0  8  0  0  0  0]
 [ 0  0  1  0  0  0  0  0  0  0  0  0  0  1  0  0  0  5  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  1  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.85      0.52      0.64        66
          1       1.00      0.44      0.62        27
          2       0.78      0.23      0.36        30
          3       0.00      0.00      0.00        14
          4       0.80      0.27      0.40        15
          5       0.96      0.57      0.72        42
          6       0.00      0.00      0.00         8
          7       0.42      0.31      0.36        16
          8       1.00      0.22      0.36         9
          9       1.00      0.20      0.33         5
         10       0.30      0.50      0.38        14
         11       0.21      0.60      0.32         5
         12       0.24      0.43      0.31        14
         13       0.45      0.79      0.57        43
         14       0.67      0.80      0.73         5
         15       0.36      1.00      0.53         5
         16       0.17      0.53      0.26        15
         17       0.25      0.71      0.37         7
         18       0.00      0.00      0.00         1
         19       0.00      0.00      0.00         2
         20       0.00      0.00      0.00         2

avg / total       0.63      0.47      0.48       345




Here is the evaluation of the model performance: 

The accuracy score is 0.466667.

The Cohen's Kappa socre is 0.420067.

The micro precistion is 0.466667, the macro precision is 0.450389.

The micro recall is 0.466667, the macro recall is 0.387268.

The micro F1 score is 0.466667, the macro F1 score is 0.345045.



## support data with ewc, highly overfitting, high regularization
### 5 class
The confusion matrix is as follows: 

[[59  1  2  0  4]
 [ 2 22  2  0  1]
 [ 5  1 22  0  2]
 [ 1  0  5  7  1]
 [ 0  1  1  0 13]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.88      0.89      0.89        66
          1       0.88      0.81      0.85        27
          2       0.69      0.73      0.71        30
          3       1.00      0.50      0.67        14
          4       0.62      0.87      0.72        15

avg / total       0.83      0.81      0.81       152




Here is the evaluation of the model performance: 

The accuracy score is 0.809211.

The Cohen's Kappa socre is 0.735001.

The micro precistion is 0.809211, the macro precision is 0.813429.

The micro recall is 0.809211, the macro recall is 0.761751.

The micro F1 score is 0.809211, the macro F1 score is 0.766388.

### 10 class
The confusion matrix is as follows: 

[[53  0  2  0  0  3  5  3  0  0]
 [ 1 17  1  1  0  2  3  1  0  1]
 [ 1  0 19  0  0  2  1  6  1  0]
 [ 0  0  6  7  0  1  0  0  0  0]
 [ 0  0  1  0  6  3  1  4  0  0]
 [ 0  2  0  0  1 36  2  0  0  1]
 [ 0  0  0  1  0  1  4  0  1  1]
 [ 1  0  1  0  2  1  0 10  0  1]
 [ 0  0  0  0  0  0  0  0  9  0]
 [ 0  0  1  0  0  0  0  0  1  3]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.95      0.80      0.87        66
          1       0.89      0.63      0.74        27
          2       0.61      0.63      0.62        30
          3       0.78      0.50      0.61        14
          4       0.67      0.40      0.50        15
          5       0.73      0.86      0.79        42
          6       0.25      0.50      0.33         8
          7       0.42      0.62      0.50        16
          8       0.75      1.00      0.86         9
          9       0.43      0.60      0.50         5

avg / total       0.75      0.71      0.72       232




Here is the evaluation of the model performance: 

The accuracy score is 0.706897.

The Cohen's Kappa socre is 0.654875.

The micro precistion is 0.706897, the macro precision is 0.647845.

The micro recall is 0.706897, the macro recall is 0.654814.

The micro F1 score is 0.706897, the macro F1 score is 0.632131.

### 15 class
The confusion matrix is as follows: 

[[39  0  0  0  0  0  0  0  0  1  7  0 14  5  0]
 [ 0 17  0  0  0  0  0  0  0  0  0  2  5  3  0]
 [ 2  0 14  0  0  0  0  1  0  0  1  0  0 12  0]
 [ 0  0  1  5  0  0  0  2  0  0  0  0  0  6  0]
 [ 0  1  0  0  2  0  0  1  0  0  0  0  2  8  1]
 [ 0  1  0  0  0 25  0  0  0  0  1  7  0  8  0]
 [ 0  1  0  0  0  1  0  0  0  0  0  1  1  4  0]
 [ 0  0  0  0  0  0  0  8  0  0  3  1  0  4  0]
 [ 0  0  0  0  0  1  0  0  2  0  1  1  0  4  0]
 [ 0  0  0  0  0  0  0  0  0  1  0  0  0  4  0]
 [ 0  0  0  0  0  0  0  0  0  0  6  0  0  8  0]
 [ 0  0  0  0  0  1  0  0  0  0  0  3  1  0  0]
 [ 1  0  1  0  0  0  0  0  0  0  0  0  8  4  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0 43  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  4]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.93      0.59      0.72        66
          1       0.85      0.63      0.72        27
          2       0.88      0.47      0.61        30
          3       1.00      0.36      0.53        14
          4       1.00      0.13      0.24        15
          5       0.89      0.60      0.71        42
          6       0.00      0.00      0.00         8
          7       0.67      0.50      0.57        16
          8       1.00      0.22      0.36         9
          9       0.50      0.20      0.29         5
         10       0.32      0.43      0.36        14
         11       0.20      0.60      0.30         5
         12       0.26      0.57      0.36        14
         13       0.38      1.00      0.55        43
         14       0.80      0.80      0.80         5

avg / total       0.73      0.57      0.57       313




Here is the evaluation of the model performance: 

The accuracy score is 0.565495.

The Cohen's Kappa socre is 0.510595.

The micro precistion is 0.565495, the macro precision is 0.644276.

The micro recall is 0.565495, the macro recall is 0.473009.

The micro F1 score is 0.565495, the macro F1 score is 0.474531.

### 21 class
The confusion matrix is as follows: 

[[29  0  0  0  0  0  0  0  0  0  6  0 12  3  0  1 12  0  1  1  1]
 [ 0  9  0  0  0  0  0  0  0  0  0  0  3  0  0  0 15  0  0  0  0]
 [ 1  0  6  0  0  0  0  1  0  0  1  0  1  9  0  3  3  4  0  0  1]
 [ 0  0  1  2  0  0  0  0  0  0  0  0  0  3  1  0  2  4  0  0  1]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  1  4  5  0  3  2  0  0  0]
 [ 0  0  0  0  0 25  0  0  0  0  0  0  0  2  2  2  9  0  1  1  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  6  0  0  1  0]
 [ 0  0  0  0  0  0  0  2  0  0  3  0  0  2  0  2  2  4  0  1  0]
 [ 0  0  0  0  0  0  0  0  1  0  1  0  0  2  0  2  0  1  2  0  0]
 [ 0  0  0  0  0  0  0  0  0  1  1  0  0  1  0  1  1  0  0  0  0]
 [ 2  0  0  0  0  0  0  0  0  0  6  0  1  1  0  1  2  0  0  0  1]
 [ 0  0  0  0  0  1  0  0  0  0  0  1  0  0  1  0  2  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  9  2  0  0  2  1  0  0  0]
 [ 0  0  0  0  1  0  0  0  0  0  0  0  0 31  0  1  6  2  1  1  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  4  0  0  1  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  4  1  0  0  0  0]
 [ 0  0  1  0  0  0  0  0  0  0  1  0  1  2  0  0 10  0  0  0  0]
 [ 0  0  1  0  0  0  0  0  0  0  0  0  0  1  0  0  0  5  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  1  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0]]



The classification result for each class is as follows: 

             precision    recall  f1-score   support

          0       0.91      0.44      0.59        66
          1       1.00      0.33      0.50        27
          2       0.67      0.20      0.31        30
          3       1.00      0.14      0.25        14
          4       0.00      0.00      0.00        15
          5       0.96      0.60      0.74        42
          6       0.00      0.00      0.00         8
          7       0.67      0.12      0.21        16
          8       1.00      0.11      0.20         9
          9       1.00      0.20      0.33         5
         10       0.32      0.43      0.36        14
         11       1.00      0.20      0.33         5
         12       0.32      0.64      0.43        14
         13       0.49      0.72      0.58        43
         14       0.29      0.80      0.42         5
         15       0.22      0.80      0.35         5
         16       0.13      0.67      0.22        15
         17       0.20      0.71      0.31         7
         18       0.00      0.00      0.00         1
         19       0.17      0.50      0.25         2
         20       0.00      0.00      0.00         2

avg / total       0.66      0.42      0.44       345




Here is the evaluation of the model performance: 

The accuracy score is 0.423188.

The Cohen's Kappa socre is 0.377567.

The micro precistion is 0.423188, the macro precision is 0.492058.

The micro recall is 0.423188, the macro recall is 0.362869.

The micro F1 score is 0.423188, the macro F1 score is 0.304074.
