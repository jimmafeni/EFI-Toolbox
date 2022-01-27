# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 06:05:37 2021
@author: Aayush Kumar
"""
from termcolor import colored
import classification_methods as clf_m
import results_gen_methods as rgm


# ---------------------------------------- Artificial Neural Network Classifier -------------------------------------#

def ann_clf(feature=None, label=None, x_train=None, x_test=None, y_train=None, y_test=None, cv=None):
    print(
        colored('<-------------------Current Model: "Artificial Neural Network Classifier" running------------------>',
                'green', attrs=['bold']))
    import warnings
    warnings.filterwarnings("ignore", category=Warning)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from silence_tensorflow import silence_tensorflow
    silence_tensorflow()
    import tensorflow as tf
    print(f'Tensorflow version {tf.__version__}')
    import keras
    from tensorflow.python.keras.layers import Dense, Dropout
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras import regularizers
    import numpy as np

    # Class details
    dft_class_names = np.unique(label)
    class_names = dft_class_names.tolist()
    class_names = map(str, class_names)
    class_names = list(class_names)
    print(f"class names,{class_names}")

    # selecting number of hidden layers:
    # Nh=Ns/(α∗(Ni+No))
    # Ni = number of input neurons.
    # No = number of output neurons.
    # Ns = number of samples in training data set.
    # α = an arbitrary scaling factor usually 2-10.

    n_i = feature.shape[1]
    n_o = len(np.unique(label))
    #n_s = label.size
    alpha = 2
    #n_h = n_s / (alpha * (n_i + n_o))
    hl_1 = n_i * alpha
    hl_2 = hl_1 * alpha
    hl_3 = hl_2 * alpha

    #  Build classifier
    def build_model():
        # using pre-defined keras sequential model
        model = tf.keras.models.Sequential()
        # defining the input layer of the Neural network and activated regularizes for L2 norm
        model.add(Dense(n_i, input_dim=n_i, activation='relu'))
        # defining the hidden layer 1 of the Neural network and activated regularizes for L2 norm
        model.add(Dense(hl_1,  activation='relu'))
        # defining the hidden layer 2 of the Neural network and activated regularizes for L2 norm
        model.add(Dense(hl_2, activation='relu'))
        # defining the hidden layer 3 of the Neural network and activated regularizes for L2 norm
        model.add(Dense(hl_3, activation='relu'))
        # defining the output layer of the Neural network i.e. output node
        model.add(Dense(n_o, activation='softmax'))
        # defining the metrics, optimizer and loss function for compiling
        loss_fc = "sparse_categorical_crossentropy"
        opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=opt, loss=loss_fc, metrics=['accuracy'])
        return model

    # Grid Search Cross Validation
    param_grid = {'epochs': [25, 50, 75, 100],
                  'batch_size': [2, 4, 8],
                  }

    # Creating Model Object and auto-tuning
    model_clf = KerasClassifier(build_fn=build_model, verbose=0)
    tuned_model = clf_m.auto_tuning(model_clf, x_train, y_train, cv, param_grid)
    # Predictions
    y_pred = tuned_model.predict(x_test)
    # results
    clf_m.get_results(tuned_model, x_test, y_test, y_pred, cv, class_names, model_name="Artificial Neural Network")
    # %% ROC-AUC Curve
    y_prob = tuned_model.predict_proba(x_test)
    y_score = y_prob
    clf_m.plot_model_roc_curve(tuned_model, y_test, y_score, class_names=dft_class_names,
                               model_name="Artificial Neural "
                                          "Network")
    print(
        colored('<------------------------"Artificial Neural Network Classifier" - Evaluation Complete - ----------->',
                'blue',
                attrs=['bold']))

    # filename = "Artificial Neural Network.h5"
    # build_model().save(os.path.join(rgm.generating_results('learned_models'), filename))
    # print("Model saved")

    return tuned_model


# ---------------------------------------- Support vector machines Classifier ------------------------------------ #

def svm_clf(feature=None, label=None, x_train=None, x_test=None, y_train=None, y_test=None, cv=None):
    print(colored('<-----------------------Current Model: "Special Vector Machine Classifier" '
                  'running------------------>', 'green',
                  attrs=['bold']))
    from sklearn.model_selection import cross_val_score
    from sklearn import svm
    from sklearn.multiclass import OneVsRestClassifier
    import numpy as np

    # Class details
    dft_class_names = np.unique(label)
    class_names = dft_class_names.tolist()
    class_names = map(str, class_names)
    class_names = list(class_names)
    print(f"class names,{class_names}")

    # Learn to predict each class against the other
    model_clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
    svm_acc = cross_val_score(model_clf, x_train, y_train, cv=cv)
    mean = svm_acc.mean()
    std = svm_acc.std()
    print(f"Base model cross-val accuracy,{mean}")
    print(f"standard deviation,{std}")

    param_grid = [
        {'estimator__kernel': ['rbf'], 'estimator__gamma': [1e-1, 1e-2, 1e-3, 1e-4],
         'estimator__C': [1, 10, 100, 1000]},
        {'estimator__kernel': ['linear'], 'estimator__C': [1, 10, 100, 1000]},
        {'estimator__kernel': ['poly'], 'estimator__gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'estimator__degree': [3, 4, 5, 6],
         'estimator__C': [1, 10, 100, 1000]},
        {'estimator__kernel': ['sigmoid'], 'estimator__gamma': [1e-1, 1e-2, 1e-3, 1e-4],
         'estimator__C': [1, 10, 100, 1000]}]

    tuned_model = clf_m.auto_tuning(model_clf, x_train, y_train, cv, param_grid)
    # Tuned Model Prediction
    y_pred = tuned_model.predict(x_test)
    # results
    clf_m.get_results(tuned_model, x_test, y_test, y_pred, cv, class_names, model_name="Support vector machines")
    # %% ROC-AUC Curve
    y_prob = tuned_model.predict_proba(x_test)
    y_score = y_prob
    print(y_score)
    clf_m.plot_model_roc_curve(tuned_model, y_test, y_score, class_names=dft_class_names,
                               model_name="Support vector machines")
    print(colored('<------------------------"Support vector machines Classifier" - Evaluation Complete - ----------->',
                  'blue',
                  attrs=['bold']))
    import pickle
    import os
    # save the model to disk
    filename = f'Support vector machines.sav'
    pickle.dump(tuned_model, open(os.path.join(rgm.generating_results('learned_models'), filename), 'wb'))

    return tuned_model


# ------------------------------------------Random Forest Classifier ------------------------------------------------ #

def random_forest_clf(feature=None, label=None, x_train=None, x_test=None, y_train=None, y_test=None, cv=None):
    print(colored('<-------------------------Current Model: "Random Forest classifier" '
                  'running-------------------------->', 'green', attrs=['bold']))
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    dft_class_names = np.unique(label)
    class_names = dft_class_names.tolist()
    class_names = map(str, class_names)
    class_names = list(class_names)
    print(f"class names,{class_names}")

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Learn to predict each class against the other
    model_clf = RandomForestClassifier(criterion='entropy')
    rf_acc = cross_val_score(model_clf, x_train, y_train, cv=cv)
    mean = rf_acc.mean()
    std = rf_acc.std()
    print(f"Base model cross-val accuracy,{mean}")
    print(f"standard deviation,{std}")

    param_grid = {
        "criterion": ['gini', 'entropy'],
        "n_estimators": [100, 200, 300],
        "min_samples_leaf": [4, 5, 6, 7, 8],
        "bootstrap": [True, False],
    }
    tuned_model = clf_m.auto_tuning(model_clf, x_train, y_train, cv, param_grid)
    y_pred = tuned_model.predict(x_test)
    # results
    clf_m.get_results(tuned_model, x_test, y_test, y_pred, cv, class_names, model_name="Random Forest classifier")
    # %% ROC-AUC Curve
    y_prob = tuned_model.predict_proba(x_test)
    y_score = y_prob
    clf_m.plot_model_roc_curve(tuned_model, y_test, y_score, class_names=dft_class_names, model_name="Random Forest")
    print(colored('<------------------------"Random Forest classifier" - Evaluation Complete - ----------->', 'blue',
                  attrs=['bold']))
    import pickle
    import os
    # save the model to disk
    filename = f'Random Forest classifier.sav'
    pickle.dump(tuned_model, open(os.path.join(rgm.generating_results('learned_models'), filename), 'wb'))
    return tuned_model


# -------------------------------------------- LightGBM Classifier ---------------------------------------------------#

def lgbm_clf(feature=None, label=None, x_train=None, x_test=None, y_train=None, y_test=None, cv=None):
    from sklearn.model_selection import cross_val_score
    import numpy as np
    import lightgbm as lgb

    print(colored('<-----------Current Model: "LightGBM Classifier" running----------->', 'green', attrs=['bold']))

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Specifying the parameter
    if len(np.unique(y_train)) == 2:
        obj = 'binary'  # Binary target feature
        metric = 'binary_logloss'  # metric for binary classification
        nclass = 1
    else:
        obj = 'multiclass'  # Multi-class target feature
        metric = 'multi_logloss'  # metric for multi-class
        nclass = len(np.unique(y_train))

    model_clf = lgb.LGBMClassifier(objective=obj, learning_rate=0.03, boosting_type='gbdt', num_class=nclass,
                                   metric=metric, n_jobs=-1)
    lgb_acc = cross_val_score(model_clf, x_train, y_train, cv=cv)
    mean = lgb_acc.mean()
    std = lgb_acc.std()
    print(f"Base model cross-val accuracy,{mean}")
    print(f"standard deviation,{std}")
    param_grid = {
        'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
        'num_leaves': [10, 12, 14, 16],
        'max_depth': [4, 5, 6, 8, 10],
        'n_estimators': [50, 60, 70, 80],
        'is_unbalance': [True]}

    tuned_model = clf_m.auto_tuning(model_clf, x_train, y_train, cv, param_grid)

    # Prediction
    y_pred = tuned_model.predict(x_test)

    # class details
    dft_class_names = np.unique(y_train)
    class_names = dft_class_names.tolist()
    class_names = map(str, class_names)
    class_names = list(class_names)

    # results
    clf_m.get_results(tuned_model, x_test, y_test, y_pred, cv, class_names, model_name="LightGBM Classifier")

    # %% ROC-AUC Curve
    y_prob = tuned_model.predict_proba(x_test)
    y_score = y_prob
    clf_m.plot_model_roc_curve(tuned_model, y_test, y_score, class_names=dft_class_names,
                               model_name="LightGBM Classifier")
    print(colored('<------------------------"LightGBM Classifier" - Evaluation Complete - ----------->', 'blue',
                  attrs=['bold']))
    import pickle
    import os
    # save the model to disk
    filename = f'LightGBM Classifier.sav'
    pickle.dump(tuned_model, open(os.path.join(rgm.generating_results('learned_models'), filename), 'wb'))

    return tuned_model


# ------------------------------------------- Logistic Regressor Classifier ------------------------------------------#

def logistic_regression_clf(feature=None, label=None, x_train=None, x_test=None, y_train=None, y_test=None, cv=None):
    print(colored('<--------------------------Current Model: "Logistic Regressor Classifier" '
                  'running------------------->', 'green', attrs=['bold']))
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Class details
    dft_class_names = np.unique(label)
    class_names = dft_class_names.tolist()
    class_names = map(str, class_names)
    class_names = list(class_names)
    print(f"class names,{class_names}")

    model_clf = LogisticRegression(solver='newton-cg', penalty='l2', max_iter=5000)
    lr_acc = cross_val_score(model_clf, x_train, y_train, cv=cv)
    mean = lr_acc.mean()
    std = lr_acc.std()
    print(f"Base model cross-val accuracy,{mean}")
    print(f"standard deviation,{std}")
    param_grid = {
        'solver': ['newton-cg', 'lbfgs', 'saga', 'sag'],
        'penalty': ['l2'],
        'C': [100, 10, 1.0, 0.1, 0.01],
    }
    tuned_model = clf_m.auto_tuning(model_clf, x_train, y_train, cv, param_grid)
    # Prediction
    y_pred = tuned_model.predict(x_test)
    # results
    clf_m.get_results(tuned_model, x_test, y_test, y_pred, cv, class_names, model_name='Logistic Regressor '
                                                                                       'Classifier')
    # %% ROC-AUC Curve
    y_prob = tuned_model.predict_proba(x_test)
    y_score = y_prob
    clf_m.plot_model_roc_curve(tuned_model, y_test, y_score, class_names=dft_class_names,
                               model_name='Logistic Regressor ')

    print(
        colored('<------------------------"Logistic Regressor Classifier" - Evaluation Complete - ----------->', 'blue',
                attrs=['bold']))
    import pickle
    import os
    # save the model to disk
    filename = f'Logistic Regressor Classifier.sav'
    pickle.dump(tuned_model, open(os.path.join(rgm.generating_results('learned_models'), filename), 'wb'))

    return tuned_model
