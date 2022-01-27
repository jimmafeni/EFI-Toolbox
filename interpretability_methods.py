# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 06:05:37 2021
@author: Aayush Kumar
"""
import results_gen_methods as rgm
import os
import intrp_tech as it


#######################################################################################################################
# ------------------------- Methods used in implementation of Interpretability Techniques --------------------------- #
#######################################################################################################################

def calculate_majority_vote(df_results, top_features_selected, title=""):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    # sorting the columns to determine feature rank - returns the indices that would sort an array.
    sorted_indices = np.argsort(-(df_results.to_numpy()), axis=0)

    array_rank = np.empty_like(sorted_indices)

    for i in range(len(array_rank[0])):
        array_rank[sorted_indices[:, i], i] = np.arange(len(array_rank[:, i]))

    # Summing the number of times a feature has been within the top n most important
    truth_table = array_rank < top_features_selected
    features_on_top = truth_table.astype(int)
    number_occurrencies_on_top = features_on_top.sum(axis=1)

    # And now plotting a heatmap showing feature importance
    df_heatmap_data = pd.DataFrame(number_occurrencies_on_top, columns=['Rank-' + title])
    df_heatmap_data.set_index(df_results.index.values, inplace=True)
    labels = np.array(df_heatmap_data.index.values)
    labels = labels.reshape((labels.shape[0], 1))
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    # Hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title, fontsize=18)
    # Remove axis
    ax.axis('off')
    sns.heatmap(df_heatmap_data, annot=labels, cmap='Greens', fmt='', ax=ax, annot_kws={"fontsize": 8},
                cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"}, linewidths=.5).set_title(title)
    file = f'{title}-rank.png'
    plt.savefig(os.path.join(rgm.generating_results('Rank'), file), dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')

    # Deleting those features that did not appear on the top (to produce a summarised figure)
    df_heatmap_data_mainfeatures = df_heatmap_data.drop(df_heatmap_data[df_heatmap_data.iloc[:, 0] < 1].index)
    labels = np.array(df_heatmap_data_mainfeatures.index.values)
    labels = labels.reshape((labels.shape[0], 1))
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    plt.title(title, fontsize=18)
    # Hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove axis
    ax.axis('off')
    sns.heatmap(df_heatmap_data_mainfeatures, annot=labels, cmap='Greens', fmt='', ax=ax, annot_kws={"fontsize": 8},
                cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"}, linewidths=.5).set_title(
        title + '- Majority Voting')
    file = f'{title}-Majority Voting.png'
    plt.savefig(os.path.join(rgm.generating_results('Majority Voting'), file), dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')
    return df_heatmap_data


def ensemble_feature_importance(shap_results=None, lime_results=None, pi_results=None, model_name=None,
                                top_feature_majority_voting=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    df_SHAP_Rank = calculate_majority_vote(shap_results, top_feature_majority_voting, title=f"Shap-{model_name}")
    df_LIME_Rank = calculate_majority_vote(lime_results, top_feature_majority_voting, title=f'LIME-{model_name}')
    df_PI_Rank = calculate_majority_vote(pi_results, top_feature_majority_voting, title=f'PI_{model_name}')
    df_ENSEMBLE_ML_MODEL_Rank = calculate_majority_vote(shap_results + lime_results + pi_results,
                                                        top_feature_majority_voting,
                                                        title=f'{model_name}')
    df_All_Feature_Importance_Rank = pd.concat([df_SHAP_Rank, df_LIME_Rank, df_PI_Rank], axis=1, sort=False)
    # Ensemble as the sum of all votes
    df_Ensemble_Majority_Vote = df_All_Feature_Importance_Rank.sum(axis=1)
    # Plotting the bar chart with the feature importance - majority voting count
    df_top_features = df_Ensemble_Majority_Vote.nlargest(15)
    df_top_features = df_top_features.sort_values(axis=0, ascending=True)
    df_top_features.plot.barh()
    plt.title(f'Ensemble_feature_imp-{model_name}')
    file = f'{model_name}-single_fusion.png'
    plt.savefig(os.path.join(rgm.generating_results('single_fusion'), file), dpi=600)
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')
    return df_Ensemble_Majority_Vote


def majority_vote_func(all_stacked, num_feats):
    import numpy as np
    from collections import Counter
    from statistics import mean
    # obtain rank of each feature for each feature importance
    all_stacked_rank = np.zeros((num_feats, 1))
    for i in range(all_stacked.shape[1]):
        all_stacked_rank = np.hstack((all_stacked_rank, np.reshape(np.argsort(all_stacked[:, i]), (-1, 1))))

    all_stacked_rank = all_stacked_rank[:, 1:]

    # select the most common rank for each feature
    most_common_rank = []
    for i in range(all_stacked_rank.shape[0]):
        majority_number = Counter(all_stacked_rank[i, :]).most_common()[0][1]
        if len(Counter(all_stacked_rank[i, :]).most_common()) > 1:
            second_majority_number = Counter(all_stacked_rank[i, :]).most_common()[1][1]
        else:
            second_majority_number = None
        # record instances if there are two majority, else just append the majority
        if (majority_number) == (second_majority_number):
            most_common_rank.append(
                [
                    int(Counter(all_stacked_rank[i, :]).most_common()[0][0]),
                    int(Counter(all_stacked_rank[i, :]).most_common()[1][0]),
                ]
            )
        else:
            most_common_rank.append(([int(Counter(all_stacked_rank[i, :]).most_common()[0][0])]))

    # average of majorty
    total_scaled_majority_vote = list()
    for i in range(all_stacked_rank.shape[0]):
        temp_majority_avg = []
        for j in range(all_stacked_rank.shape[1]):
            # single majority
            if len(most_common_rank[i]) == 1:
                if all_stacked_rank[i, j] == most_common_rank[i][0]:
                    temp_majority_avg.append(all_stacked[i, j])
            # double majority
            else:
                if (all_stacked_rank[i, j] == most_common_rank[i][0]) or (
                        all_stacked_rank[i, j] == most_common_rank[i][1]):
                    temp_majority_avg.append(all_stacked[i, j])
        total_scaled_majority_vote.append(mean(temp_majority_avg))
    total_scaled_majority_vote = np.array(total_scaled_majority_vote)
    return total_scaled_majority_vote


def helper_rank_func(method, dataframe):
    from scipy.stats import kendalltau, spearmanr
    import numpy as np
    corr = dataframe.corr(method=method)

    def kendall_pval(x, y):
        return kendalltau(x, y)[1]

    def spearmanr_pval(x, y):
        return spearmanr(x, y)[1]
        # removing highly correlated values

    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) <= 0.5:
                if columns[j]:
                    columns[j] = False
    selected_columns_cr = dataframe.columns[columns]
    dft = dataframe[selected_columns_cr]
    # remvoving columns with p-values > 0.05
    p_value = 0
    if method == 'kendall':
        p_value = dft.corr(method=kendall_pval)
    if method == 'spearman':
        p_value = dft.corr(method=spearmanr_pval)

    columns = np.full((p_value.shape[0],), True, dtype=bool)
    for i in range(p_value.shape[0]):
        for j in range(i + 1, p_value.shape[0]):
            if p_value.iloc[i, j] > 0.05:
                if columns[j]:
                    columns[j] = False
    selected_columns_pvr = dft.columns[columns]
    dftp = dft[selected_columns_pvr]
    dftp["final"] = dftp.mean(axis=1)
    return np.array(dftp["final"])

# -------------------------------- Interpretability Techniques Implementation ---------------------------------------- #


def intpr_technqs_impl(feature, label, model, test_data_size, model_name):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=test_data_size, random_state=42,
                                                        shuffle=True, stratify=label)
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return it.shap_model(x_test, model, model_name), it.lime_model(x_test, y_test, model, model_name), it.perm_imp(
        x_test,
        y_test, model,
        model_name)


# Implementation for fuzzy logic

def fuzzy_intpr_impl(feature, label, model, model_name, fcv):
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    import numpy as np
    FINAL_FUZZY = pd.DataFrame()
    skf = StratifiedKFold(n_splits=fcv, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(feature, label):
        x_train_fold, x_test_fold = feature.iloc[train_index], feature.iloc[test_index]
        y_train_fold, y_test_fold = label.iloc[train_index], label.iloc[test_index]
        x_train_ip = x_train_fold.reset_index(drop=True)
        y_train_ip = y_train_fold.reset_index(drop=True)
        x_test_ip = x_test_fold.reset_index(drop=True)
        y_test_ip = y_test_fold.reset_index(drop=True)

        fuzzy_temp = pd.DataFrame(np.vstack(
            [it.shap_model(x_test_ip, model, model_name), it.lime_model(x_test_ip, y_test_ip, model, model_name)
                , it.perm_imp(x_test_ip, y_test_ip, model, model_name)]), columns=[model_name])
        FINAL_FUZZY = FINAL_FUZZY.append(fuzzy_temp)

    return FINAL_FUZZY.values.flatten().tolist()
