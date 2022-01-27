# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 06:05:37 2021
@author: Aayush Kumar
"""

import data_preprocessing as dp
import pandas as pd
import os
import interpretability_methods as im
import classification_methods as clf_m
import fuzzy_logic as fl
import results_gen_methods as rgm
import user_xi as usxi
from sklearn.model_selection import train_test_split
import classification_models as cm
import multi_fusion as mf

"""
set experiment configuration before running the tool
"""
######################################################################################################################
# STAGE 1 : Experiments configuration
######################################################################################################################

param, model_selected = usxi.exp_config_portal()

# # Loading and Preprocessing Dataset

x, y = dp.data_preprocessing(param[0], param[1])

print('Features:', x)

print('Class:', y.value_counts())

# Train/Test Data Size, for evaluation of classification models

data_size_for_testing = param[2] / 100

print(f"Your data_size_for_testing:{data_size_for_testing}")

# Data size to be used evaluation of interpretability

data_size_for_interpretability = param[3] / 100

print(f"Your data_size_for_interpretability:{data_size_for_interpretability}")

# K-Fold Cross validation for model
cv = param[4]

print(f"Your cross - validation folds:  {cv}")

# K-Fold Cross validation for Fuzzy model

fcv = param[5]

print(f"Your cross - validation folds:  {fcv}")

######################################################################################################################
# STAGE 2 : Model configuration for the classification pipeline / Initialize dataframe for storing evaluation results
######################################################################################################################

model_selected = model_selected
SHAP_RESULTS = pd.DataFrame(index=x.columns.values, columns=usxi.models_to_eval)
LIME_RESULTS = pd.DataFrame(index=x.columns.values, columns=usxi.models_to_eval)
PI_RESULTS = pd.DataFrame(index=x.columns.values, columns=usxi.models_to_eval)
ENSEMBLE_ML_MODEL = pd.DataFrame(index=x.columns.values, columns=['SHAP', 'LIME', 'PI'])
FUZZY_DATA = pd.DataFrame()

######################################################################################################################
# STAGE 3 : Classification model Evaluation based on the Model configuration and models selected
#####################################################################################################################


def generate_fi(model_selected, x, y, data_size_for_testing,data_size_for_interpretability):
    # split into train test sets as per the configuration
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=data_size_for_testing, random_state=42,
                                                        shuffle=True,
                                                        stratify=y)
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)


    for model_name in model_selected:
        if model_name == "LightGBM Classifier":
            SHAP_RESULTS[model_name], LIME_RESULTS[model_name], PI_RESULTS[
                model_name] = im.intpr_technqs_impl(x, y,
                                                    cm.lgbm_clf(x, y, x_train, x_test, y_train, y_test, cv),
                                                    data_size_for_interpretability,
                                                    model_name)

        elif model_name == "Logistic Regressor classifier":
            SHAP_RESULTS[model_name], LIME_RESULTS[model_name], PI_RESULTS[
                model_name] = im.intpr_technqs_impl(x, y,
                                                    cm.logistic_regression_clf(x, y, x_train, x_test, y_train, y_test, cv),
                                                    data_size_for_interpretability,
                                                    model_name)


        elif model_name == "Artificial Neural Network":

            SHAP_RESULTS[model_name], LIME_RESULTS[model_name], PI_RESULTS[
                model_name] = im.intpr_technqs_impl(x, y,
                                                    cm.ann_clf(x, y, x_train, x_test, y_train, y_test, cv),
                                                    data_size_for_interpretability,
                                                    model_name)

        elif model_name == "Random Forest classifier":
            SHAP_RESULTS[model_name], LIME_RESULTS[model_name], PI_RESULTS[
                model_name] = im.intpr_technqs_impl(x, y,
                                                    cm.random_forest_clf(x, y, x_train, x_test, y_train,y_test, cv),
                                                    data_size_for_interpretability,
                                                    model_name)

        elif model_name == 'Support vector machines':
            SHAP_RESULTS[model_name], LIME_RESULTS[model_name], PI_RESULTS[
                model_name] = im.intpr_technqs_impl(x, y,
                                                    cm.svm_clf(x, y, x_train, x_test, y_train, y_test, cv),
                                                    data_size_for_interpretability,
                                                    model_name)



SHAP_RESULTS, LIME_RESULTS, PI_RESULTS = generate_fi(model_selected, x, y, data_size_for_testing,data_size_for_interpretability)
######################################################################################################################
# STAGE 4 : MODEL SPECIFIC Feature Importance -  Single Fusion
######################################################################################################################

SHAP_RESULTS.dropna(how='all', axis=1)
LIME_RESULTS.dropna(how='all', axis=1)
PI_RESULTS.dropna(how='all', axis=1)

SHAP_RESULTS.to_excel(os.path.join(rgm.generating_results('Results_XLS'), "SHAP_RESULTS.xlsx"))
LIME_RESULTS.to_excel(os.path.join(rgm.generating_results('Results_XLS'), "LIME_RESULTS.xlsx"))
PI_RESULTS.to_excel(os.path.join(rgm.generating_results('Results_XLS'), "PI_RESULTS.xlsx"))

for model_name in model_selected:

    if model_name == "LightGBM Classifier":
        ENSEMBLE_ML_MODEL['PI'] = PI_RESULTS[model_name]
        ENSEMBLE_ML_MODEL['LIME'] = LIME_RESULTS[model_name]
        ENSEMBLE_ML_MODEL['SHAP'] = SHAP_RESULTS[model_name]
        im.ensemble_feature_importance(ENSEMBLE_ML_MODEL[['SHAP']], ENSEMBLE_ML_MODEL[['LIME']],
                                       ENSEMBLE_ML_MODEL[['PI']], model_name,
                                       top_feature_majority_voting=int((len(x.columns.values) * 0.50)))

        FUZZY_DATA["LightGBM Classifier"] = im.fuzzy_intpr_impl(x, y, clf_m.load_models("LightGBM Classifier"),
                                                                data_size_for_interpretability,
                                                                model_name, fcv)
        ENSEMBLE_ML_MODEL.to_excel(os.path.join(rgm.generating_results('Results_XLS'), f"{model_name}.xlsx"))

    elif model_name == "Logistic Regressor classifier":
        ENSEMBLE_ML_MODEL['PI'] = PI_RESULTS[model_name]
        ENSEMBLE_ML_MODEL['LIME'] = LIME_RESULTS[model_name]
        ENSEMBLE_ML_MODEL['SHAP'] = SHAP_RESULTS[model_name]
        im.ensemble_feature_importance(ENSEMBLE_ML_MODEL[['SHAP']], ENSEMBLE_ML_MODEL[['LIME']],
                                       ENSEMBLE_ML_MODEL[['PI']], model_name,
                                       top_feature_majority_voting=int((len(x.columns.values) * 0.50)))

        FUZZY_DATA["Logistic Regressor classifier"] = im.fuzzy_intpr_impl(x, y, clf_m.load_models(
            "Logistic Regressor classifier"),
                                                                          data_size_for_interpretability,
                                                                          model_name, fcv)

        ENSEMBLE_ML_MODEL.to_excel(os.path.join(rgm.generating_results('Results_XLS'), f"{model_name}.xlsx"))

    elif model_name == "Artificial Neural Network":
        ENSEMBLE_ML_MODEL['PI'] = PI_RESULTS[model_name]
        ENSEMBLE_ML_MODEL['LIME'] = LIME_RESULTS[model_name]
        ENSEMBLE_ML_MODEL['SHAP'] = SHAP_RESULTS[model_name]
        im.ensemble_feature_importance(ENSEMBLE_ML_MODEL[['SHAP']], ENSEMBLE_ML_MODEL[['LIME']],
                                       ENSEMBLE_ML_MODEL[['PI']], model_name,
                                       top_feature_majority_voting=int((len(x.columns.values) * 0.50)))

        FUZZY_DATA["Artificial Neural Network"] = im.fuzzy_intpr_impl(x, y,
                                                                      cm.ann_clf(x, y, x_train, x_test, y_train, y_test,
                                                                                 cv),
                                                                      data_size_for_interpretability,
                                                                      model_name, fcv)
        ENSEMBLE_ML_MODEL.to_excel(os.path.join(rgm.generating_results('Results_XLS'), f"{model_name}.xlsx"))

    elif model_name == "Random Forest classifier":
        ENSEMBLE_ML_MODEL['PI'] = PI_RESULTS[model_name]
        ENSEMBLE_ML_MODEL['LIME'] = LIME_RESULTS[model_name]
        ENSEMBLE_ML_MODEL['SHAP'] = SHAP_RESULTS[model_name]
        im.ensemble_feature_importance(ENSEMBLE_ML_MODEL[['SHAP']], ENSEMBLE_ML_MODEL[['LIME']],
                                       ENSEMBLE_ML_MODEL[['PI']], model_name,
                                       top_feature_majority_voting=int((len(x.columns.values) * 0.50)))

        FUZZY_DATA["Random Forest classifier"] = im.fuzzy_intpr_impl(x, y,
                                                                     clf_m.load_models("Random Forest classifier"),
                                                                     data_size_for_interpretability,
                                                                     model_name, fcv)
        ENSEMBLE_ML_MODEL.to_excel(os.path.join(rgm.generating_results('Results_XLS'), f"{model_name}.xlsx"))
    elif model_name == "Support vector machines":
        ENSEMBLE_ML_MODEL['PI'] = PI_RESULTS[model_name]
        ENSEMBLE_ML_MODEL['LIME'] = LIME_RESULTS[model_name]
        ENSEMBLE_ML_MODEL['SHAP'] = SHAP_RESULTS[model_name]
        im.ensemble_feature_importance(ENSEMBLE_ML_MODEL[['SHAP']], ENSEMBLE_ML_MODEL[['LIME']],
                                       ENSEMBLE_ML_MODEL[['PI']], model_name,
                                       top_feature_majority_voting=int((len(x.columns.values) * 0.50)))

        FUZZY_DATA["Support vector machines"] = im.fuzzy_intpr_impl(x, y, clf_m.load_models("Support vector machines"),
                                                                    data_size_for_interpretability,
                                                                    model_name, fcv)
        ENSEMBLE_ML_MODEL.to_excel(os.path.join(rgm.generating_results('Results_XLS'), f"{model_name}.xlsx"))

FUZZY_DATA.dropna(how='all', axis=1)
FUZZY_DATA.to_excel(os.path.join(rgm.generating_results('Results_XLS'), "FUZZY-DATA_Before_AC.xlsx"))

######################################################################################################################
# STAGE 5:  Feature Importance -  Multi Fusion
######################################################################################################################


mf.multi_fusion_feature_imp(SHAP_RESULTS, LIME_RESULTS, PI_RESULTS, x, model_selected)

######################################################################################################################
# STAGE 6:  Feature Importance -  FUZZY Fusion
######################################################################################################################

fl.fuzzy_implementation(FUZZY_DATA, model_selected)

######################################################################################################################
# STAGE 7 :  Generate Report
######################################################################################################################

# <-------------------------------------------- Evaluation Specific Reports ------------------------------------------>

# Machine learning model evaluation
rgm.generate_eval_report('perm')
rgm.generate_eval_report('ROC')
rgm.generate_eval_report('Confusion_Matrix')
# Interpretability Techniques
rgm.generate_eval_report("Permutation Importances")
rgm.generate_eval_report("Lime")
rgm.generate_eval_report("SHAP")
# Fusion Approach
rgm.generate_eval_report('single_fusion')
rgm.generate_eval_report('Multi-Fusion')
rgm.generate_eval_report('Majority Voting')
rgm.generate_eval_report('Rank')
# Fuzzy Approach
rgm.generate_eval_report('FUZZY')

# <------------------------------------------------ Model Specific Reports ------------------------------------------->

for model_name in model_selected:
    if model_name == "LightGBM Classifier":
        rgm.generate_model_report("LightGBM Classifier")
    elif model_name == "Logistic Regressor classifier":
        rgm.generate_model_report("Logistic Regressor classifier")
    elif model_name == "Artificial Neural Network":
        rgm.generate_model_report("Artificial Neural Network")
    elif model_name == "Random Forest classifier":
        rgm.generate_model_report("Random Forest classifier")
    elif model_name == 'Support vector machines':
        rgm.generate_model_report('Support vector machines')

# <------------------------------------------------ Multi Fusion Report ---------------------------------------------->

rgm.generate_multi_fusion(model_selected)
