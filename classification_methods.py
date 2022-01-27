# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 06:05:37 2021
@author: Aayush Kumar
"""
import results_gen_methods as rgm
import os
from termcolor import colored

#######################################################################################################################
# ------------------------- Methods used in implementation of Classification Techniques ----------------------------- #
#######################################################################################################################


def auto_tuning(model, x_train, y_train, k_fold, param_grid):
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(estimator=model, refit=True, n_jobs=-1, cv=k_fold, param_grid=param_grid)
    grid_model = grid.fit(x_train, y_train, )  # Fitting the GridSearch Object on the Train Set
    print("Best: %f using %s" % (grid_model.best_score_, grid_model.best_params_))
    # %% Model Tuning- Building a Tuned Model with Best Parameters
    # Creating Tuned Model Object with KerasClassifier
    tuned_model = grid_model.best_estimator_
    return tuned_model


def get_results(model, feature, y_test, y_pred, n_split, class_names, model_name):
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np
    import pandas as pd
    import dataframe_image as dfi
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold

    performance = pd.DataFrame()
    # K-fold accuracy scores
    kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)
    results_acc = cross_val_score(model, feature, y_test, cv=kfold, scoring='accuracy')
    # K-fold accuracy scores
    print('K-fold Cross Validation Accuracy Results: ', results_acc)
    # K-fold f1 scores
    results_f1 = cross_val_score(model, feature, y_test, cv=kfold, scoring="f1_weighted")
    print('K-fold Cross Validation f1_weighted Results: ', results_f1)
    # Classification Report
    model_report = classification_report(y_test, y_pred, target_names=class_names)

    # Confusion Matrix
    model_conf = confusion_matrix(y_test, y_pred)
    fig = ConfusionMatrixDisplay(confusion_matrix=model_conf, display_labels=class_names)
    fig.plot(cmap="Blues")
    plt.title(f'Confusion_Matrix-{model_name}')
    clf_cm = f'{model_name}-cm.png'
    plt.savefig(os.path.join(rgm.generating_results('Confusion_Matrix'), clf_cm), dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')
    FP = model_conf.sum(axis=0) - np.diag(model_conf)
    FN = model_conf.sum(axis=1) - np.diag(model_conf)
    TP = np.diag(model_conf)
    TN = model_conf.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    print(
        colored('<------------------------------"over-all performance of the model"--------------------------------->',
                'yellow', attrs=['bold']))
    performance['class'] = class_names
    # K-fold accuracy
    print(f'Mean accuracy for-{model_name}         :{results_acc.mean()}')
    performance['K-Fold mean accuracy']: results_acc.mean()
    # K-fold f1 scores
    print(f'Mean f1_weighted score for-{model_name}:{results_f1.mean()}')
    performance['K-Fold mean 1_weighted score']: results_f1.mean()
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    print(f'Sensitivity for-{model_name}           :{TPR}')
    performance['Sensitivity'] = TPR
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    print(f'Specificity for-{model_name}           :{TNR}')
    performance['Specificity'] = TNR
    # Classification report
    print(f"classification report for-{model_name} :\n{model_report}")
    model_report_df = pd.DataFrame(
        classification_report(y_test, y_pred, target_names=class_names, output_dict=True)).transpose()
    model_report_df.reset_index(inplace=True)
    # Confusion Matrix
    print(f"confusion Matrix for-{model_name}      :\n{model_conf}")
    # Summary of evaluation
    print(performance)
    model_report_df['Model_Evaluated'] = model_name
    model_eval_df = pd.concat([model_report_df, performance], ignore_index=False, axis=1)
    print(model_eval_df)
    model_eval_df_styled = model_eval_df.style.background_gradient()
    fn = f'{model_name}-perm.png'
    dfi.export(model_eval_df_styled, os.path.join(rgm.generating_results('perm'), fn))

    print(
        colored(
            '<---------------------------------------------------------------------------------------------------->',
            'yellow', attrs=['bold']))

# inspired from kaggle.com : https://www.kaggle.com/nirajvermafcb/comparing-various-ml-models-roc-curve-comparison

def plot_model_roc_curve(model, y_test, y_score, class_names, model_name=''):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    # noinspection PyUnresolvedReferences
    from scipy import interp
    import numpy as np
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(class_names)
    if n_classes == 2:
        if hasattr(model, 'predict_proba'):
            prb = y_score
            if prb.shape[1] > 1:
                y_score = prb[:, prb.shape[1] - 1]
            else:
                y_score = y_score.ravel()

        else:
            raise AttributeError("Estimator doesn't have a probability or confidence scoring system!")
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve (area = {0:3.2%})'.format(roc_auc), linewidth=2.5)

    elif n_classes > 2:
        y_test = label_binarize(y_test, classes=class_names)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot ROC curves
        plt.figure(figsize=(6, 4))
        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:2.2%})'
                                                   ''.format(roc_auc["micro"]), linewidth=3)

        plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:2.2%})'
                                                   ''.format(roc_auc["macro"]), linewidth=3)

        for i, label in enumerate(class_names):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:2.2%})'
                                           ''.format(label, roc_auc[i]), linewidth=2, linestyle=':')
        roc_auc = roc_auc["macro"]
    else:
        raise ValueError('Number of classes should be atleast 2 or more')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve-{model_name}')
    plt.legend(loc="lower right")
    roc_fn = f'{model_name}-roc.png'
    plt.savefig(os.path.join(rgm.generating_results('ROC'), roc_fn), dpi=600, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')
    return roc_auc


def load_models(x):
    import pickle
    from keras.models import load_model
    import os
    current_directory = os.getcwd()
    if x == "Artificial Neural Network":
        rd = os.path.join(current_directory, r"results", r"learned_models", rf'{x}.h5')
        loaded_model = load_model(rd)
    else:
        rd = os.path.join(current_directory, r"results", r"learned_models", rf'{x}.sav')
        # load the model from disk
        loaded_model = pickle.load(open(rd, 'rb'))
    return loaded_model

