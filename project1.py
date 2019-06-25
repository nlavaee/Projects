# EECS 445 - Fall 2018
# Project 1 - project1.py

import pandas as pd
import numpy as np
import itertools
import string
import math

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt

from helper import *


def generate_feature_vector(df):
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.
    
    Args:
        df: pd.Dataframe, with columns [Time, Variable, Value]
    
    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {'Age': 32, 'Gender': 0, 'mean_HR': 84, ...}
    """


    data_frame = df
    feature_dict = {}
    static_variables = config['static']
    timeseries_variables = config['timeseries']

#     - Age
# - Gender
# - Height
# - ICUType
# - Weight


    #data_frame[data_frame < 0] = np.nan
    data_frame['Value'] = data_frame['Value'].replace([-1], np.nan)

    #if one of the variables does not exist in the dataframe make it np.nan
    for elem in timeseries_variables:
        if (data_frame['Variable'].str.contains(elem).any()) == False:
            feature_dict["mean_"+elem] = np.nan

    data_frame = data_frame.groupby('Variable', as_index = False)['Value'].mean()

    data_frame['Variable'] = 'mean_' + data_frame['Variable'].astype(str)

    feature_dict.update(dict(zip(data_frame['Variable'], data_frame['Value'])))

    feature_dict.pop("mean_MechVent", None)

    feature_dict['Age'] = feature_dict.pop('mean_Age')
    feature_dict['Gender'] = feature_dict.pop('mean_Gender')
    feature_dict['Height'] = feature_dict.pop('mean_Height')
    feature_dict['ICUType'] = feature_dict.pop('mean_ICUType')
    feature_dict['Weight'] = feature_dict.pop('mean_Weight')


    return feature_dict



def generate_challenge_feature_vector(df):
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.
    
    Args:
        df: pd.Dataframe, with columns [Time, Variable, Value]
    
    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {'Age': 32, 'Gender': 0, 'mean_HR': 84, ...}
    """


    data_frame = df
    feature_dict = {}
    static_variables = config['static']
    timeseries_variables = config['timeseries']

#     - Age
# - Gender
# - Height
# - ICUType
# - Weight


    #data_frame[data_frame < 0] = np.nan
    data_frame['Value'] = data_frame['Value'].replace([-1], np.nan)

    #if one of the variables does not exist in the dataframe make it np.nan
    for elem in timeseries_variables:
        if (data_frame['Variable'].str.contains(elem).any()) == False:
            feature_dict["median_"+elem] = np.nan

    data_frame = data_frame.groupby('Variable', as_index = False)['Value'].median()

    data_frame['Variable'] = 'median_' + data_frame['Variable'].astype(str)

    feature_dict.update(dict(zip(data_frame['Variable'], data_frame['Value'])))

    feature_dict['Age'] = feature_dict.pop('median_Age')
    feature_dict['Gender'] = feature_dict.pop('median_Gender')
    feature_dict['Height'] = feature_dict.pop('median_Height')
    feature_dict['ICUType'] = feature_dict.pop('median_ICUType')
    feature_dict['Weight'] = feature_dict.pop('median_Weight')


    return feature_dict

def impute_missing_values(X):
    """
    For each feature column, impute missing values  (np.nan) with the 
    population mean for that feature.
    
    Args:
        X: np.array, shape (N, d). X could contain missing values
    Returns:
        X: np.array, shape (N, d). X does not contain any missing values
    """
    # TODO: implement this function

    column_means = np.nanmean(X, axis=0)

    for i, column in enumerate(X.T):
        for k, j in enumerate(column):
            if np.isnan(j):
                column[k] = column_means[i]


    return X


def impute_challenge_missing_values(X):
    """
    For each feature column, impute missing values  (np.nan) with the 
    population median for that feature.
    
    Args:
        X: np.array, shape (N, d). X could contain missing values
    Returns:
        X: np.array, shape (N, d). X does not contain any missing values
    """
    # TODO: implement this function

    column_medians = np.nanmedian(X, axis=0)

    for i, column in enumerate(X.T):
        for k, j in enumerate(column):
            if np.isnan(j):
                column[k] = column_medians[i]
                #column[k] = 0

    return X


def normalize_feature_matrix(X):
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: np.array, shape (N, d).
    Returns:
        X: np.array, shape (N, d). Values are normalized per column.
    """
    # TODO: implement this function
    for column in X.T:
        min_value = np.min(column)
        max_value = np.max(column)
        for k, j in enumerate(column):
            if min_value == max_value:
                column[k] = 1.
            else:
                column[k] = float((j - min_value)/(max_value - min_value))
    return X


def get_classifier(kernel='linear', penalty='l2', C=1.0, gamma=0.0, class_weight=None):
    """
    Return a linear/rbf kernel SVM classifier based on the given
    penalty function and regularization parameter C.
    """
    # TODO: Optionally implement this helper function if you would like to
    # instantiate your SVM classifiers in a single function. You will need
    # to use the above parameters throughout the assignment.


    
    raise NotImplementedError


def performance(clf_trained, X, y, metric='accuracy'):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted scores from clf_trained and X.
    Input:
        clf_trained: a fitted instance of sklearn estimator
        X : (n,d) np.array containing features
        y_true: (n,) np.array containing true labels
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance measure as a float
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.

    tn, fp, fn, tp = metrics.confusion_matrix(y, clf_trained.predict(X)).ravel()

    if metric == 'accuracy':
        return metrics.accuracy_score(y, clf_trained.predict(X))

    if metric == 'f1-score':
        # precision = tp/(tp + fp)
        # recall = tp/(tp + fn)

        if tp == 0 and (fp == 0 or fn == 0):
            return 0.0

        else:
            return metrics.f1_score(y, clf_trained.predict(X))

    if metric == 'auroc':
        return metrics.roc_auc_score(y, clf_trained.decision_function(X))

    if metric == 'precision':
        
        if tp == 0 and fp == 0:
            return 0.0

        else:
            return metrics.precision_score(y, clf_trained.predict(X))

    if metric == 'sensitivity':
        #tn, fp, fn, tp = metrics.confusion_matrix(y, clf_trained.predict(X)).ravel()
        if tp == 0 and fn == 0:
            return 0.0

        else:
            return float(tp)/(tp + fn)

    if metric == 'specificity':
        #tn, fp, fn, tp = metrics.confusion_matrix(y, clf_trained.predict(X)).ravel()

        if fp == 0 and tn == 0:
            return 0.0

        else:
            return float(tn)/(fp + tn)


def cv_performance(clf, X, y, metric='accuracy', k=5):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    
    ## HINT: You may find the StratifiedKFold from sklearn.model_selection
    ## to be useful

    # Put the performance of the model on each fold in the scores array


    scores = []

    skf = StratifiedKFold(n_splits=k, shuffle=False, random_state=None)

    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train, "TEST:", test)
       

        #train and classify on each of these sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scores.append(performance(clf.fit(X_train, y_train), X_test, y_test, metric))

    

    # And return the average performance across all fold splits.
    return np.array(scores).mean()


def select_param_linear(X, y, metric='accuracy', k=5, C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    #HINT: You should be using your cv_performance function here
    #to evaluate the performance of each SVM

    metric_values = []
 
    for i in C_range:
        metric_values.append(cv_performance(SVC(kernel='linear', C=i), X, y, metric, k))
        print(metric_values)
        
    max_C_index = np.argmax(metric_values)


    return C_range[max_C_index], np.max(metric_values)


def plot_weight(X, y, penalty, *metric, C_range):
    """
    Takes as input the training data X and labels y and plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier.
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    norm0 = []


    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2-penalty, degree=1 SVM to the data (X, y)

    for i in C_range:

        if penalty == 'l2':
            clf = SVC(kernel='linear', C=i)
        elif penalty == 'l1':
            clf = LinearSVC(C=i, penalty='l1', loss='squared_hinge', dual=False)

        clf.fit(X, y)

        norm0.append(np.count_nonzero(clf.coef_))


   
    # This code will plot your L0-norm as a function of C
    plt.plot(C_range, norm0)
    plt.xscale('log')
    plt.legend(['L0-norm'])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title('Norm-'+penalty+'_penalty.png')
    plt.savefig('Norm-'+penalty+'_penalty.png')
    plt.close()


def select_param_rbf(X, y, metric='accuracy', k=5, param_range=[]):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        parameter_values: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for gamma. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter value(s) for a RBF-kernel SVM that maximize
        the average 5-fold CV performance
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...



    metric_values = []

    for i in range(len(param_range)):
        for j in range(len(param_range[i])):
            metric_values.append(cv_performance(SVC(kernel='rbf', C = param_range[i][j][0], gamma=param_range[i][j][1]), X, y, metric, k))
            print(metric_values)

        
    max_param_index = np.argmax(metric_values)

    row_max = math.floor(max_param_index / 7)
    column_max = max_param_index % 7

    #print(row_max, column_max, max_param_index)

    return param_range[row_max][column_max], np.max(metric_values)

def select_param_poly(X, y, metric='auroc', k=5, param_range = []):

    max_value = float("-inf")
    current = 0
    param_indices = []
    for i in range(len(param_range)):
        for j in range(len(param_range[i])):
            for h in range(len(param_range[i][j])):
                current = (cv_performance(SVC(kernel='poly', C = param_range[i][j][h][0], gamma = param_range[i][j][h][1], degree = param_range[i][j][h][2]), X, y, metric, k))
                print(current)
                print(param_range[i][j][h][0], param_range[i][j][h][1], param_range[i][j][h][2])
                if max_value < current:
                    max_value = current
                    param_indices = [i, j, h]
    #max_param_index = np.argmax(metric_values)

    #row_max = math.floor(max_param_index / 7)
    #column_max = max_param_index % 7

    #print(row_max, column_max, max_param_index)

    C = param_indices[0]
    gamma = param_indices[1]
    degree = param_indices[2]

    return param_range[C][gamma][degree], max_value




def main():
    #Read data
    # #NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       #IMPLEMENTING generate_feature_vector, fill_missing_values AND normalize_feature_matrix
    X_train, y_train, X_test, y_test, feature_names = get_train_test_split()

    # TODO: Questions 1, 2, 3


    print("average feature vector ", np.mean(X_train, 0))
    print("dimensionality of feature vector ", X_train.shape)
    print("feature names: ", feature_names)

    C_range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]


    print(" ")
    print("part c")
    print(" ")
    print(select_param_linear(X_train, y_train, metric='accuracy', k=5, C_range = C_range, penalty='l2'))
    print(select_param_linear(X_train, y_train, metric='f1-score', k=5, C_range = C_range, penalty='l2'))
    print(select_param_linear(X_train, y_train, metric='auroc', k=5, C_range = C_range, penalty='l2'))
    print(select_param_linear(X_train, y_train, metric='precision', k=5, C_range = C_range, penalty='l2')) 
    print(select_param_linear(X_train, y_train, metric='sensitivity', k=5, C_range = C_range, penalty='l2'))
    print(select_param_linear(X_train, y_train, metric='specificity', k=5, C_range = C_range, penalty='l2'))
    print(" ")

    clf = SVC(kernel='linear', C=100.0)

    print("part d")
    print(" ")
    print(performance(clf.fit(X_train, y_train), X_test, y_test, metric ='accuracy'))
    print(performance(clf.fit(X_train, y_train), X_test, y_test, metric ='f1-score'))
    print(performance(clf.fit(X_train, y_train), X_test, y_test, metric ='auroc'))
    print(performance(clf.fit(X_train, y_train), X_test, y_test, metric ='precision'))
    print(performance(clf.fit(X_train, y_train), X_test, y_test, metric ='sensitivity'))
    print(performance(clf.fit(X_train, y_train), X_test, y_test, metric ='specificity'))
    print(" ")
   

   	#2.1e
    plot_weight(X_train, y_train, 'l2', C_range= C_range)


     #2.1f
    
    coeffs = []

    clf = SVC(kernel='linear', C=1.0)

    clf.fit(X_train, y_train)

    coeffs = clf.coef_

    idx = coeffs.argsort()

    for i in idx[0][:4]:
        print("Negative Coeff: ", coeffs[0][i], " Feature Name: ", feature_names[i])


    for i in idx[0][-4:]:
        print("Positive Coeff: ", coeffs[0][i], " Feature Name: ", feature_names[i])

    #2.2

    param_range = np.zeros((7, 7, 2))
    gamma = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    for i in range(len(C_range)):
        for j in range(len(gamma)):
            param_range[i][j] = [C_range[i], gamma[j]]

    print("RBF-Kernel SVM ", select_param_rbf(X_train, y_train, metric='auroc', k=5, param_range = param_range))

    #2.3a

    metric_values = []

    for i in C_range:
        metric_values.append(cv_performance(LinearSVC(C=i, penalty='l1', loss='squared_hinge', dual=False), X_train, y_train, metric='auroc', k=5))

    max_C_index = np.argmax(metric_values)

    print("Linear_Kernel SVM ", C_range[max_C_index], np.max(metric_values))


    #2.3b
    plot_weight(X_train, y_train,'l1', C_range = C_range)



    #3
    print("part 3.1")
    performance_measures = ['accuracy', 'f1-score', 'auroc', 'precision', 'sensitivity', 'specificity']
    clf = SVC(kernel='linear', C=1.0, class_weight={-1: 1, 1: 50})

    for i in performance_measures:
        print(performance(clf.fit(X_train, y_train), X_test, y_test, metric=i))


    print("part 3.2")
    numb_positives = (y_train > 0).sum()
    numb_negatives = (y_train < 0).sum()

    print("Numb of positives and negatives " , numb_positives, numb_negatives)

    #WP = 6, WN = 1

    clf = SVC(kernel='linear', C=1.0, class_weight={-1: 1, 1: 6})
    for i in performance_measures:
        print("cv_performance on wn and wp ", cv_performance(clf, X_train, y_train, metric=i, k=5))


    for i in performance_measures:
        print("performance measures on wn and wp ", performance(clf.fit(X_train, y_train), X_test, y_test, metric=i))



    print("part 3.3")

    clf = SVC(kernel='linear', C=1.0, class_weight={-1: 1, 1: 1})

    y_score = clf.fit(X_train, y_train).decision_function(X_test)
    n_classes = [-1, 1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)

    clf = SVC(kernel='linear', C=1.0, class_weight={-1: 1, 1: 5})
    y_score = clf.fit(X_train, y_train).decision_function(X_test)

    fpr1 = dict()
    tpr1 = dict()
    roc_auc1 = dict()
    fpr1, tpr1, _ = metrics.roc_curve(y_test, y_score)
    roc_auc1 = metrics.auc(fpr1, tpr1)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='green', lw=lw, label='ROC Curve (area = %0.2f)' % roc_auc)
    plt.plot(fpr1, tpr1, color='blue', lw=lw, label='ROC Curve (area = %0.2f)' % roc_auc1)
    plt.plot([0, 1], [0, 1], color='purple', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()





    #Read challenge data
    #TODO: Question 4: Apply a classifier to heldout features, and then use
          #generate_challenge_labels to print the predicted labels
    X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()

    X_train, X_test, y_train, y_test = train_test_split(X_challenge, y_challenge, test_size=0.2)



    print("average feature vector ", np.mean(X_challenge, 0))
    print(feature_names)


    #treat numerical and categorical variables differently
    #consider other summary statistics for numerical variables --> use median for impute
    #best way to improve auroc and f1-score

    #changed the feature vector to look at medians and place missing values with the median
    #kept normalize the same

    #check for weights of positive to negative for the data

    numb_positives = (y_challenge > 0).sum()
    numb_negatives = (y_challenge < 0).sum()

    print("Numb of positives and negatives " , numb_positives, numb_negatives)


    


    C_range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    param_range = np.zeros((7, 7, 2))
	gamma = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]


    C_poly_range = [1e-2, 1e-1, 1, 1e1, 1e2]
    gamma_poly = [1e-1, 1, 1e1]
    degrees = [2, 3]
    param_range_degrees = np.zeros((5,3,2,3))

    for i in range(len(C_range)):
        for j in range(len(gamma)):
            param_range[i][j] = [C_range[i], gamma[j]]

    
    for i in range(len(C_poly_range)):
        for j in range(len(gamma_poly)):
            for k in range(len(degrees)):
                param_range_degrees[i][j][k] = [C_range[i], gamma_poly[j], degrees[k]]


    


    performance_measures = ['auroc', 'f1-score', 'accuracy', 'precision', 'sensitivity', 'specificity']

    #what produces best parameter for polynomial and rbf kernel

    print("The best C and degree are ", select_param_poly(X_train, y_train, metric='auroc', k=5, param_range = param_range_degrees))
    print("the best c and gamma for auroc are " , select_param_rbf(X_train, y_train, metric='auroc', k=5, param_range=param_range))
    print("the best c, degrees, and gamma is ", select_param_poly(X_train, y_train, metric='f1-score', k=5, param_range=param_range_degrees))
    print("the best c, degrees, and gamma is ", select_param_poly(X_train, y_train, metric='auroc', k=5, param_range=param_range_degrees))
    print("best C for linear kernal ", select_param_linear(X_train, y_train, metric='auroc', k=5, C_range=C_range))
    print("best C and gamma for f1-score are ", select_param_rbf(X_train, y_train, metric='f1-score', k=5, param_range= param_range))



    #clf = SVC(kernel='poly', C=0.01, degree=2, gamma=10, class_weight={-1:1, 1:5})

    clf_linear = SVC(kernel='linear', C=.10,  class_weight={-1: 1, 1: 5})

    clf_rbf = SVC(kernel = 'rbf', C = 1000, gamma = .1, class_weight={-1:1, 1:5})

    	
    test the performance of our model on our training data that we had

    print("performance of this on C is ", performance(clf.fit(X_train, y_train), X_test, y_test))

   	print("performance of this on C is ", performance(clf_linear.fit(X_train, y_train), X_test, y_test))

   	print("performance of this on C is ", performance(clf_rbf.fit(X_train, y_train), X_test, y_test))


    clf_trained = clf.fit(X_challenge, y_challenge)
   

    for i in performance_measures:
    	
    	print("the cv_performance for ", i, " is ", cv_performance(clf, X_challenge, y_challenge, metric = i, k=5))
    	print("the performance for ", i, " is ", performance(clf_trained, X_challenge, y_challenge, metric=i))
        print("cv_performance on wn and wp ", cv_performance(clf, X_train, y_train, metric=i, k=5))
        

    tn, fp, fn, tp = metrics.confusion_matrix(y_challenge, clf_trained.predict(X_challenge), labels=[-1,1]).ravel()
    print(tn, fp, fn, tp)

    ylabel = int(float(clf_trained.predict(X_heldout)))
    yscore = clf_trained.decision_function(X_heldout)
    generate_challenge_labels(int(float(ylabel)), int(float(yscore)), 'nlavaee')




if __name__ == '__main__':
    main()
