def fuzzy_implementation(x_data,fuzzy_data, model_selected):
    import numpy as np
    import skfuzzy as fuzz
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    import interpretability_methods as im
    import results_gen_methods as rgm
    from skfuzzy import control as ctrl
    from sklearn.model_selection import train_test_split

    # Initializing varibales
    dwhole_GB = None
    dwhole_LR = None
    dwhole_ANN = None
    dwhole_RF = None
    dwhole_SVM = None
    GB = None
    LR = None
    ANN = None
    RF = None
    SVM = None
    #number of feature
    num_feat = fuzzy_data.shape[1]
    
    # import feature importance data.

    df_whole = fuzzy_data

    # calculating the actual coefficient, through majority voting method

    df_whole["AC"] = im.majority_vote_func(np.array(df_whole), len(df_whole))

    df_whole.to_excel(os.path.join(rgm.generating_results('Results_XLS'), "FUZZY-DATA.xlsx"))

    print(df_whole)

    # split into train and test sets

    df_train, df_test = train_test_split(df_whole, test_size=0.1, random_state=42)

    # inter quatile

    df_whole_qtl = df_train.quantile([0, 0.25, 0.5, 0.75, 1])

    # Define the Universe

    # Create membership functions
    col_names = df_whole.columns

    print(col_names)

    for model_name in model_selected:

        if model_name == "LightGBM Classifier":

            dwhole_GB = np.arange(df_train["LightGBM Classifier"].min(), df_train["LightGBM Classifier"].max(), 0.001)

            GB = ctrl.Antecedent(dwhole_GB, "LightGBM Classifier")

            # Gradient boost
            GB["low"] = fuzz.zmf(dwhole_GB, df_whole_qtl["LightGBM Classifier"][0.00],
                                 df_whole_qtl["LightGBM Classifier"][0.50])
            GB['moderate'] = fuzz.trimf(dwhole_GB, [df_whole_qtl["LightGBM Classifier"][0.25],
                                                    df_whole_qtl["LightGBM Classifier"][0.50],
                                                    df_whole_qtl["LightGBM Classifier"][0.75]])
            GB['high'] = fuzz.smf(dwhole_GB, df_whole_qtl["LightGBM Classifier"][0.50],
                                  df_whole_qtl["LightGBM Classifier"][1.00])

        elif model_name == "Logistic Regressor classifier":

            dwhole_LR = np.arange(df_train["Logistic Regressor classifier"].min(), df_train["Logistic Regressor "
                                                                                            "classifier"].max(), 0.001)

            LR = ctrl.Antecedent(dwhole_LR, "Logistic Regressor classifier")

            # "Logistic Regressor classifier"
            LR['low'] = fuzz.zmf(dwhole_LR, df_whole_qtl["Logistic Regressor classifier"][0.00],
                                 df_whole_qtl["Logistic Regressor classifier"][0.50])
            LR['moderate'] = fuzz.trimf(dwhole_LR,
                                        [df_whole_qtl["Logistic Regressor classifier"][0.25], df_whole_qtl["Logistic "
                                                                                                           "Regressor"
                                                                                                           " classifier"][
                                            0.50],
                                         df_whole_qtl["Logistic Regressor classifier"][0.75]])
            LR['high'] = fuzz.smf(dwhole_LR, df_whole_qtl["Logistic Regressor classifier"][0.50],
                                  df_whole_qtl["Logistic Regressor classifier"][1.00])

        elif model_name == "Artificial Neural Network":

            dwhole_ANN = np.arange(df_train["Artificial Neural Network"].min(),
                                   df_train["Artificial Neural Network"].max(), 0.001)

            ANN = ctrl.Antecedent(dwhole_ANN, "Artificial Neural Network")

            # "Artificial Neural Network"
            ANN['low'] = fuzz.zmf(dwhole_ANN, df_whole_qtl["Artificial Neural Network"][0.00],
                                  df_whole_qtl["Artificial Neural Network"][0.50])
            ANN['moderate'] = fuzz.trimf(dwhole_ANN,
                                         [df_whole_qtl["Artificial Neural Network"][0.25],
                                          df_whole_qtl["Artificial Neural Network"][0.50],
                                          df_whole_qtl["Artificial Neural Network"][0.75]])
            ANN['high'] = fuzz.smf(dwhole_ANN, df_whole_qtl["Artificial Neural Network"][0.50],
                                   df_whole_qtl["Artificial Neural Network"][1.00])

        elif model_name == "Random Forest classifier":

            dwhole_RF = np.arange(df_train["Random Forest classifier"].min(),
                                  df_train["Random Forest classifier"].max(), 0.001)

            RF = ctrl.Antecedent(dwhole_RF, "Random Forest classifier")

            # Random forest
            RF['low'] = fuzz.zmf(dwhole_RF, df_whole_qtl["Random Forest classifier"][0.00],
                                 df_whole_qtl["Random Forest classifier"][0.50])
            RF['moderate'] = fuzz.trimf(dwhole_RF, [df_whole_qtl["Random Forest classifier"][0.25],
                                                    df_whole_qtl["Random Forest classifier"][0.50],
                                                    df_whole_qtl["Random Forest classifier"][0.75]])
            RF['high'] = fuzz.smf(dwhole_RF, df_whole_qtl["Random Forest classifier"][0.50],
                                  df_whole_qtl["Random Forest classifier"][1.00])

        elif model_name == "Support vector machines":

            dwhole_SVM = np.arange(df_train["Support vector machines"].min(), df_train["Support vector machines"].max(),
                                   0.001)

            SVM = ctrl.Antecedent(dwhole_SVM, "Support vector machines")

            # Support vector
            SVM['low'] = fuzz.zmf(dwhole_SVM, df_whole_qtl["Support vector machines"][0.00], df_whole_qtl["Support "
                                                                                                          "vector "
                                                                                                          "machines"][
                0.50])
            SVM['moderate'] = fuzz.trimf(dwhole_SVM,
                                         [df_whole_qtl["Support vector machines"][0.25], df_whole_qtl["Support vector "
                                                                                                      "machines"][
                                             0.50],
                                          df_whole_qtl["Support vector machines"][0.75]])
            SVM['high'] = fuzz.smf(dwhole_SVM, df_whole_qtl["Support vector machines"][0.50], df_whole_qtl["Support "
                                                                                                           "vector "
                                                                                                           "machines"][
                1.00])

    # Actual coefficient
    dwhole_AC = np.arange(df_train["AC"].min(), df_train["AC"].max(), 0.001)

    AC = ctrl.Consequent(dwhole_AC, 'AC')
    # Actual coefficients
    AC['low'] = fuzz.zmf(dwhole_AC, df_whole_qtl["AC"][0.00], df_whole_qtl["AC"][0.50])
    AC['moderate'] = fuzz.trimf(dwhole_AC, [df_whole_qtl["AC"][0.25], df_whole_qtl["AC"][0.50],
                                            df_whole_qtl["AC"][0.75]])
    AC['high'] = fuzz.smf(dwhole_AC, df_whole_qtl["AC"][0.50], df_whole_qtl["AC"][1.00])

    def rule_cal(uni, mf1, mf2, mf3, val):
        # print('value is '+ str(val))
        low = fuzz.interp_membership(uni, mf1.mf, val)
        # print(low)
        med = fuzz.interp_membership(uni, mf2.mf, val)
        # print (med)
        high = fuzz.interp_membership(uni, mf3.mf, val)
        # print (high)

        if low > med:
            if low > high:
                # print('final is low')
                return 'low'
            else:
                # print('final is high')
                return 'high'
        elif med > high:
            # print('final is med')
            return 'moderate'
        else:
            # print('final is high')
            return 'high'

    val1 = None
    val2 = None
    val3 = None
    val4 = None
    val5 = None

    rules_columns = ['GB', 'LR', 'ANN', 'RF', 'SVM']

    rules = pd.DataFrame(columns=rules_columns)

    for i in range(len(df_train)):
        obs1 = df_train.iloc[i]
        for model in model_selected:
            if model == "LightGBM Classifier":
                val1 = rule_cal(dwhole_GB, GB['low'], GB['moderate'], GB['high'], obs1["LightGBM Classifier"])
            elif model == "Logistic Regressor classifier":
                val2 = rule_cal(dwhole_LR, LR['low'], LR['moderate'], LR['high'], obs1["Logistic Regressor classifier"])
            elif model == "Artificial Neural Network":
                val3 = rule_cal(dwhole_ANN, ANN['low'], ANN['moderate'], ANN['high'], obs1["Artificial Neural Network"])
            elif model == "Random Forest classifier":
                val4 = rule_cal(dwhole_RF, RF['low'], RF['moderate'], RF['high'], obs1["Random Forest classifier"])
            elif model == "Support vector machines":
                val5 = rule_cal(dwhole_SVM, SVM['low'], SVM['moderate'], SVM['high'], obs1["Support vector machines"])

        val6 = rule_cal(dwhole_AC, AC['low'], AC['moderate'], AC['high'], obs1["AC"])

        rules = rules.append({'GB': val1, 'LR': val2, 'ANN': val3, 'RF': val4, 'SVM': val5, "AC": val6},
                             ignore_index=True)
    print(rules)
    # Display rules
    rules.dropna(how='all', axis=1, inplace=True)
    # count occurrence of rules->records
    rules = rules.groupby(rules.columns.tolist()).size().reset_index().rename(columns={0: 'records'})
    rules = rules.sort_values(by=['records'], ascending=False)
    # remove conflicting rules
    rules = rules.drop_duplicates(subset=rules.columns.tolist()[:-1], keep='first')
    rules_tf = rules.drop('records', axis=1)
    print(rules_tf)

    # Automatically generate rules
    b = None
    k = rules_tf.columns.tolist()
    lines = []
    prefix = []
    suffix = []
    first = 'ctrl.Rule('
    last = ")"
    p = ["AC"]
    for j in range(len(rules_tf)):
        for i in range(len(k) - 1):
            prefix.append(str(k[i]) + str([rules_tf.iloc[j][k[i]]]))
        c = (str(p[0]) + str([rules_tf.iloc[j, -1]]))
        c = "," + c
        d = f",label = 'rule {j + 1}'"
        e = c + d
        suffix.append(e)
        for c in range(0, len(prefix), len(k) - 1):
            chunk = prefix[c:c + len(k) - 1]
            a = "&".join(chunk)
            b = str(a)
            e = suffix[j]
        lines.append(first + b + e + last)
    res = lines

    for x in res:
        print(x, end=' ')

    # Inference systems; Mamdani inference

    rules = []
    for i in range(0, len(res)):
        rules.append(eval(res[i]))

    factor_ctrl = ctrl.ControlSystem(rules)
    importance = ctrl.ControlSystemSimulation(factor_ctrl)

    # Calculate MAE on test dataset

    def mae(true, pred):
        d = np.abs(pred - true)
        length = len(d)
        err = np.sum(d) / length
        return err

    def rmse(true, pred):
        d_squared = (pred - true) ** 2
        length = len(d_squared)
        err = 0
        for i in range(length):
            err += d_squared[i]
        err = np.sqrt(err / length)
        return err

    def sd(true, pred):
        d = (pred - true)
        s = np.std(d)

        return s

    # list to store predictions
    pred = []

    # test model performance using test dataset

    for i in range(len(df_test)):
        obs1 = df_test.iloc[i]
        for model in model_selected:
            if model == "LightGBM Classifier":
                importance.input["LightGBM Classifier"] = obs1["LightGBM Classifier"]
            elif model == "Logistic Regressor classifier":
                importance.input["Logistic Regressor classifier"] = obs1["Logistic Regressor classifier"]
            elif model == "Artificial Neural Network":
                importance.input["Artificial Neural Network"] = obs1["Artificial Neural Network"]
            elif model == "Random Forest classifier":
                importance.input["Random Forest classifier"] = obs1["Random Forest classifier"]
            elif model == 'Support vector machines':
                importance.input['Support vector machines'] = obs1['Support vector machines']
        importance.compute()
        pred.append(importance.output['AC'])
    print(pred)
    # Evaluation metrics

    error = mae(df_test.AC.values, np.asarray(pred))

    print(f"error: {error}")

    rme = rmse(df_test.AC.values, np.asarray(pred))

    print(f"error: {rme}")

    st = sd(df_test.AC.values, np.asarray(pred))

    print(f"error: {st}")

    for model in model_selected:
        if model == "LightGBM Classifier":
            GB.view(sim=importance)
            fuzzy_file = f'{model}-FUZZY-.png'
            plt.savefig(os.path.join(rgm.generating_results("FUZZY"), fuzzy_file), dpi=300)
            plt.show(block=False)
            plt.pause(3)
            plt.close('all')
        elif model == "Logistic Regressor classifier":
            LR.view(sim=importance)
            fuzzy_file = f'{model}-FUZZY.png'
            plt.savefig(os.path.join(rgm.generating_results("FUZZY"), fuzzy_file), dpi=300)
            plt.show(block=False)
            plt.pause(3)
            plt.close('all')
        elif model == "Artificial Neural Network":
            ANN.view(sim=importance)
            fuzzy_file = f'{model}-FUZZY.png'
            plt.savefig(os.path.join(rgm.generating_results("FUZZY"), fuzzy_file), dpi=300)
            plt.show(block=False)
            plt.pause(3)
            plt.close('all')
        elif model == "Random Forest classifier":
            RF.view(sim=importance)
            fuzzy_file = f'{model}-FUZZY.png'
            plt.savefig(os.path.join(rgm.generating_results("FUZZY"), fuzzy_file), dpi=300)
            plt.show(block=False)
            plt.pause(3)
            plt.close('all')
        elif model == 'Support vector machines':
            SVM.view(sim=importance)
            fuzzy_file = f'{model}-FUZZY.png'
            plt.savefig(os.path.join(rgm.generating_results("FUZZY"), fuzzy_file), dpi=300)
            plt.show(block=False)
            plt.pause(3)
            plt.close('all')

    AC.view(sim=importance)
    fuzzy_ac = f'Actual-Coefficient-FUZZY.png'
    plt.savefig(os.path.join(rgm.generating_results("FUZZY"), fuzzy_ac), dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')
    
    #funtion to extract FI for feature x
    def extract_fi(data, feat, num_feat):
        return data.loc[range(feat-1,len(data),num_feat)] 

    for i in range(num_feat):
        df_feat1 = extract_fi(df_whole,i+1,num_feat)
        fused_fi = []
        #fuse fi of multiple ML methods
        for j in range(len(df_feat1)):
                obs1 = df_feat1.iloc[j]
                for model in model_selected:
                    if model == "LightGBM Classifier":
                        importance.input["LightGBM Classifier"] = obs1["LightGBM Classifier"]
                    elif model == "Logistic Regressor classifier":
                        importance.input["Logistic Regressor classifier"] = obs1["Logistic Regressor classifier"]
                    elif model == "Artificial Neural Network":
                        importance.input["Artificial Neural Network"] = obs1["Artificial Neural Network"]
                    elif model == "Random Forest classifier":
                        importance.input["Random Forest classifier"] = obs1["Random Forest classifier"]
                    elif model == 'Support vector machines':
                        importance.input['Support vector machines'] = obs1['Support vector machines']
                importance.compute()
                fused_fi.append(importance.output['AC'])
            
    feat_qtl = np.quantile(fused_fi,[0,0.25, 0.5, 0.75,1])
    #universe of features fi
    feat_uni = np.arange(min(fused_fi),max(fused_fi),0.00001)
    #Membership 
    feat_mf = ctrl.Consequent(feat_uni, 'Importance of '+ x_data.columns[i])
    
    #fuzzy sets
    #Gradient boost
    feat_mf['low'] = fuzz.zmf(feat_uni, min(feat_qtl[0],df_whole_qtl["AC"][0.00]),
                         max(df_whole_qtl["AC"][0.50],feat_qtl[2]))
    feat_mf['moderate'] = fuzz.trimf(feat_uni, [min(df_whole_qtl["AC"][0.25],feat_qtl[1]), df_whole_qtl["AC"][0.50],
                                    max(df_whole_qtl["AC"][0.75],feat_qtl[3])])
    feat_mf['high'] = fuzz.smf(feat_uni,min(df_whole_qtl["AC"][0.50],feat_qtl[2]), 
                          max(df_whole_qtl["AC"][1.00],feat_qtl[4]))
    
    feat_mf.view()
    
    plt.savefig('feat'+str(i+1)+'.png')