def multi_fusion_feature_imp(SHAP_RESULTS, LIME_RESULTS, PI_RESULTS, feature, model_selected):
    import plotly.express as px
    import plotly.graph_objs as go
    import os
    import pandas as pd
    import numpy as np
    from scipy import stats
    from scipy.optimize import minimize
    import interpretability_methods as im
    import results_gen_methods as rgm

    feature_names = feature.columns.values
    gb = pd.DataFrame()
    lr = pd.DataFrame()
    dnn = pd.DataFrame()
    rf = pd.DataFrame()
    svm = pd.DataFrame()

    for model_name in model_selected:
        if model_name == "LightGBM Classifier":

            gb["gb_sv"] = SHAP_RESULTS['LightGBM Classifier']
            gb["gb_lm"] = LIME_RESULTS['LightGBM Classifier']
            gb["gb_pi"] = PI_RESULTS['LightGBM Classifier']

        elif model_name == "Logistic Regressor classifier":
            lr["lr_sv"] = SHAP_RESULTS["Logistic Regressor classifier"]
            lr["lr_lm"] = LIME_RESULTS["Logistic Regressor classifier"]
            lr["lr_pi"] = PI_RESULTS["Logistic Regressor classifier"]

        elif model_name == "Artificial Neural Network":
            dnn["dnn_sv"] = SHAP_RESULTS["Artificial Neural Network"]
            dnn["dnn_lm"] = LIME_RESULTS["Artificial Neural Network"]
            dnn["dnn_pi"] = PI_RESULTS["Artificial Neural Network"]

        elif model_name == "Random Forest classifier":
            rf["rf_sv"] = SHAP_RESULTS["Random Forest classifier"]
            rf["rf_lm"] = LIME_RESULTS["Random Forest classifier"]
            rf["rf_pi"] = PI_RESULTS["Random Forest classifier"]

        elif model_name == 'Support vector machines':
            svm["svm_sv"] = SHAP_RESULTS['Support vector machines']
            svm["svm_lm"] = LIME_RESULTS['Support vector machines']
            svm["svm_pi"] = PI_RESULTS['Support vector machines']

    total_all_stacked_df = pd.concat([gb, lr, dnn, rf, svm], axis=1, sort=False)
    total_all_stacked_df.dropna(how='all', axis=1)
    total_all_stacked = np.array(total_all_stacked_df)
    col_names = total_all_stacked_df.columns.values.tolist()
    total_all_stacked_df.to_excel(os.path.join(rgm.generating_results('Results_XLS'), f"total_all_stacked.xlsx"))

    ########################################################################################################################
    # mean
    ########################################################################################################################

    total_scaled_mean = np.mean(total_all_stacked, axis=1)

    ########################################################################################################################
    # median
    ########################################################################################################################

    total_scaled_median = np.median(total_all_stacked, axis=1)

    ########################################################################################################################
    # Mode
    ########################################################################################################################
    def mode_ensemble(stacked):
        total_scaled_mode = np.array([])
        for i in range(stacked.shape[0]):
            params = stats.norm.fit(stacked[i, :])

            def your_density(x):
                return -stats.norm.pdf(x, *params)

            total_scaled_mode = np.append(total_scaled_mode, minimize(your_density, 0).x[0])

        return np.reshape(total_scaled_mode, (-1,))
        
    total_scaled_mode = mode_ensemble(total_all_stacked)

    ########################################################################################################################
    # box_whiskers
    ########################################################################################################################

    def box_whiskers(stacked):
        total_scaled_baw = np.array([])
        for j in range(stacked.shape[0]):
            temp_whiskers = np.array([])
            q3 = np.quantile(stacked[j, :], 0.75)
            q1 = np.quantile(stacked[j, :], 0.25)
            upper_whiskers = q3 + (1.5 * (q3 - q1))
            lower_whiskers = q1 - (1.5 * (q3 - q1))
            for i in range(stacked[j, :].shape[0]):
                if (stacked[j, :][i] >= lower_whiskers) and (stacked[j, :][i] <= upper_whiskers):
                    temp_whiskers = np.append(temp_whiskers, stacked[j, :][i])
            total_scaled_baw = np.append(total_scaled_baw, temp_whiskers.mean())
        return total_scaled_baw

    total_scaled_box_whiskers = box_whiskers(total_all_stacked)

    ########################################################################################################################
    # Thompson Tau
    ########################################################################################################################
    # (1) calculate sample mean
    # (2) calculate delta_min = |mean - min| and delta_max|mean - max|
    # (3) tau value from tau table value for sample size 7: 1.7110
    # (4) calculate standard deviation
    # (5) multiply tau with standard deviation = tau*std threshold
    # (6) compare (3) and (5)

    tau = 1.7110

    def tau_test(test_data):
        for i in range(test_data.shape[0]):
            test_data_mean = test_data.mean()
            test_data_std = np.std(test_data, ddof=1)
            test_data_min = min(test_data)
            test_data_min_index = np.argmin(test_data)
            test_data_max = max(test_data)
            test_data_max_index = np.argmax(test_data)
            test_data_min_delta = np.abs(test_data_min - test_data_mean)
            test_data_max_delta = np.abs(test_data_max - test_data_mean)

            if (test_data_min_delta >= test_data_max_delta) and (test_data_min_delta > tau * test_data_std):
                test_data = np.delete(test_data, test_data_min_index)
            else:
                if test_data_max_delta > (tau * test_data_std):
                    test_data = np.delete(test_data, test_data_max_index)
        return test_data

    # tau_test
    total_scaled_tau_test = np.array([])
    for i in range(total_all_stacked.shape[0]):
        mean_tau = np.array([tau_test(total_all_stacked[i, :]).mean()])
        total_scaled_tau_test = np.append(total_scaled_tau_test, mean_tau)

    total_scaled_tau_test = np.array([total_scaled_tau_test])
    total_scaled_tau_test = np.reshape(total_scaled_tau_test, (-1,))

    ########################################################################################################################
    # majority vote
    ########################################################################################################################

    NUM_FEATS = len(feature_names)
    total_scaled_majority_vote = im.majority_vote_func(total_all_stacked, NUM_FEATS)

    # ######################################################################################################################
    # # kendall tau
    # ######################################################################################################################

    total_scaled_kendall_tau = im.helper_rank_func('kendall', total_all_stacked_df)


    ########################################################################################################################
    # # spearman
    ########################################################################################################################

    total_scaled_spearman = im.helper_rank_func('spearman', total_all_stacked_df)

########################################################################################################################
# display result in plots
########################################################################################################################


    methods = np.array(["Mode", "Median", "Mean", "Box-Whiskers", "Tau Test", "Majority Vote", "RATE-Kendall Tau",
                        "RATE-Spearman Rho"])
    multiplied_importance = [total_scaled_mode[:].ravel(), total_scaled_median[:].ravel(), total_scaled_mean[:].ravel(),
                             total_scaled_box_whiskers[:].ravel(), total_scaled_tau_test[:].ravel(),
                             total_scaled_majority_vote[:].ravel(), total_scaled_kendall_tau[:].ravel(),
                             total_scaled_spearman[:].ravel()]

    tmi = np.array(list(np.concatenate(multiplied_importance).flat))
    tmf = np.tile(feature_names, int(tmi.shape[0] / len(feature_names)))
    tmm = np.repeat(methods, int(tmi.shape[0] / methods.shape[0]))

    tmi_ = tmi.tolist()
    tmf_ = np.array(tmf).tolist()
    tmm_ = np.array(tmm).tolist()

    results = list(zip(tmi_, tmf_, tmm_))

    total_df_results = pd.DataFrame(results, columns=["Importance", "Features", "Methods"])
    print(total_df_results)

    # <-------------------------------------------------- PLOT - I ------------------------------------------------------>

    fig = px.bar(total_df_results, x="Features", y="Importance", color="Methods", animation_frame="Methods",
                 hover_name="Features",
                 hover_data=['Features', "Importance"], range_y=[0, 1])

    fig.update_layout(
        template="plotly_white", title="Multi-Fusion ensemble for feature importance", width=1500, height=800, )

    fig.update_traces(marker=dict(size=12, line=dict(width=2, color="Black")), selector=dict(mode="markers"),
                      opacity=0.7)

    for i in range(8):
        fig_ = go.Figure(fig.frames[i].data, fig.layout)
        pt = f'Multi_Fusion_FI_Method-{i + 1}.png'
        fig_.write_image(os.path.join(rgm.generating_results('Multi-Fusion'), pt))

    ht = f'Multi-Fusion Feature Importance.html'
    fig.write_html(os.path.join(rgm.generating_results('Multi-Fusion'), ht), auto_open=True)
    fig.show()

    # <-----------s-------------------------------------- PLOT - II ------------------------------------------------------>

    fig2 = px.box(total_df_results, x="Methods", y="Importance", color="Methods",
                  hover_data=["Importance"], range_y=[0, 1])

    fig2.update_layout(
        template="plotly_white", title="Multi-Fusion ensemble for feature importance", width=1500, height=800, )
    pt = 'Multi_Fusion_Methods.png'
    ht = f'Multi-Fusion-Methods.html'
    fig2.write_html(os.path.join(rgm.generating_results('Multi-Fusion'), ht), auto_open=True)
    fig2.write_image(os.path.join(rgm.generating_results('Multi-Fusion'), pt))
    fig2.show()

    total_df_results.to_excel(os.path.join(rgm.generating_results('Results_XLS'), f"Multi_Fusion-FI.xlsx"))

    return total_df_results
