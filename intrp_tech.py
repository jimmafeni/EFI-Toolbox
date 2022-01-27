# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 06:05:37 2021
@author: Aayush Kumar
"""
import os
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from termcolor import colored
import results_gen_methods as rgm


# ---------------------------------------------- SHAP Analysis ------------------------------------------------------- #

def shap_model(feature, model, model_name):
    print(model_name)
    if model_name in ['Random Forest classifier', 'LightGBM Classifier']:
        print(
            colored('<------------------------Current Model: "shap.TreeExplainer initiated " ----------------------->',
                    'green', attrs=['bold']))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature)
        tm = np.array(shap_values)[0]
        result = np.abs(tm).mean(0)
        result_df = pd.DataFrame(result, columns=[model_name])
        result_df = result_df.set_index(feature.columns.values)

        # Model agnostic SHAP
    else:
        print(colored('<----------------------Current Model: "shap.KernelExplainer initiated " --------------------->',
                      'green', attrs=['bold']))
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(feature, 10))
        shap_values = explainer.shap_values(feature)
        tm = np.array(shap_values)
        result = np.abs(tm).mean(0)
        result_df = pd.DataFrame(result, columns=[model_name])
        result_df = result_df.set_index(feature.columns.values)

    rt_df = result_df
    rt_df_norm = rt_df.copy()
    rt_df_norm[model_name] = (rt_df_norm[model_name] - rt_df_norm[model_name].min()) / (
            rt_df_norm[model_name].max() - rt_df_norm[model_name].min())
    print(f'shap values dataframe_norm:\n {rt_df_norm}')
    print('Expected Value: ', explainer.expected_value)
    shap.summary_plot(shap_values, feature, show=False)
    plt.title(f'shap evaluation for {model_name}')
    shap_file = f'{model_name}-SHAP.png'
    plt.savefig(os.path.join(rgm.generating_results("SHAP"), shap_file), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')
    print(colored('<--------------------------------"SHAP" - Analysis Complete ------------------------------------->',
                  'blue', attrs=['bold']))
    return rt_df_norm


# ---------------------------------------------- LIME Analysis ------------------------------------------------------- #

def lime_model(feature, label, model, model_name):
    import random
    print(
        colored('<------------------------------------Evaluation: "Lime initiated " --------------------------------->',
                'green', attrs=['bold']))
    global local_prediction, global_prediction, lime_df_norm, fig
    results_df = pd.DataFrame()
    print(model_name)
    # Class details
    dft_class_names = np.unique(label)
    class_names = dft_class_names.tolist()
    class_names = map(str, class_names)
    class_names = list(class_names)
    print(f"class names:{class_names}")

    if len(class_names) == 2:
        explainer = lime.lime_tabular.LimeTabularExplainer(feature.values, feature_names=feature.columns,
                                                           class_names=class_names, mode="classification")
        feature = feature.reset_index(drop=True)
        j = random.randint(1, int(len(feature)/2))
        selected_instance = feature.loc[[j]].values[0]
        selected_class = label.loc[[j]].values[0]
        exp = explainer.explain_instance(selected_instance, model.predict_proba,
                                         num_features=len(feature.columns), top_labels=len(class_names))
        exp.show_in_notebook(show_table=True)
        exp.save_to_file(os.path.join(rgm.generating_results("Lime"), 'lime_binary.html'))
        results_df = pd.DataFrame(exp.as_map()[0])
        lime_df = results_df.sort_values(by=[0], ascending=True)
        lime_df.drop(columns=lime_df.columns[0], axis=1, inplace=True)
        lime_df.set_index(feature.columns.values, inplace=True)
        lime_df.columns = [model_name]
        lime_df_norm = lime_df.copy()
        lime_df_norm[model_name] = (lime_df_norm[model_name] - lime_df_norm[model_name].min()) / (
                lime_df_norm[model_name].max() - lime_df_norm[model_name].min())
        local_prediction = exp.local_pred
        global_prediction = exp.predict_proba
        fig = exp.as_pyplot_figure()
        plt.title(f"Local interpretation of randomly generated instance of class-{selected_class}")
        print(lime_df_norm)
    if len(class_names) > 2:
        explainer = lime.lime_tabular.LimeTabularExplainer(feature.values, feature_names=feature.columns,
                                                           class_names=class_names, mode="classification")
        feature = feature.reset_index(drop=True)
        j = random.randint(1, int(len(feature)/2))
        selected_instance = feature.loc[[j]].values[0]
        selected_class = label.loc[[j]].values[0]
        exp = explainer.explain_instance(selected_instance, model.predict_proba,
                                         num_features=len(feature.columns), top_labels=len(class_names))
        exp.show_in_notebook(show_table=True)
        exp.save_to_file(os.path.join(rgm.generating_results("Lime"), 'lime_multiclass.html'))
        for i in range(0, len(class_names)):
            exp_df = pd.DataFrame(exp.as_map()[i])
            results_df = results_df.append(exp_df.sort_values(by=[0], ascending=True), ignore_index=True)
        local_prediction = exp.local_pred
        global_prediction = exp.predict_proba
        fig = exp.as_pyplot_figure()
        plt.title(f"Local interpretation of randomly generated instance of class-{selected_class}")
        lime_df = results_df.pivot_table([1], columns=[0], aggfunc='mean').transpose()
        lime_df.set_index(feature.columns.values, inplace=True)
        lime_df.columns = [model_name]
        lime_df_norm = lime_df.copy()
        lime_df_norm[model_name] = (lime_df_norm[model_name] - lime_df_norm[model_name].min()) / (
                lime_df_norm[model_name].max() - lime_df_norm[model_name].min())
        print(lime_df_norm)
    print("Explanation Local Prediction              : ", local_prediction)
    print("Explanation Global Prediction Probability : ", global_prediction)
    lime_file = f'{model_name}-LIME.png'
    fig.savefig(os.path.join(rgm.generating_results("Lime"), lime_file), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')
    print(colored('<----------------------------------------"LIME" - Analysis Complete ------------------------------>',
                  'blue', attrs=['bold']))
    return lime_df_norm


# ------------------------------------------- Permutation Importance Analysis ---------------------------------------- #

def perm_imp(feature, label, model, model_name):
    print(
        colored('<--------------------------------------Evaluation: "PI initiated " -------------------------------->',
                'green', attrs=['bold']))
    from sklearn.inspection import permutation_importance
    print(model_name)
    result = permutation_importance(model, feature, label, n_repeats=10,
                                    random_state=42, scoring='accuracy')
    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=feature.columns[sorted_idx])
    ax.set_title("Permutation Importances")
    plt.title(f'Permutation Importances for {model_name}')
    fig.tight_layout()
    pi_file = f'{model_name}-PI.png'
    plt.savefig(os.path.join(rgm.generating_results("Permutation Importances"), pi_file), dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')
    PI_df = pd.DataFrame(result.importances_mean, columns=[model_name], index=feature.columns.values)
    PI_df_norm = PI_df.copy()
    PI_df_norm[model_name] = (PI_df_norm[model_name] - PI_df_norm[model_name].min()) / (
            PI_df_norm[model_name].max() - PI_df_norm[model_name].min())
    print(PI_df_norm)
    print(
        colored('<-----------------------"Permutation Importance" - Analysis Complete ------------------------------>',
                'blue', attrs=['bold']))
    return PI_df_norm



