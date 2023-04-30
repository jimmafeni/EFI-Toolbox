# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 06:05:37 2021
@author: Aayush Kumar
"""

import pandas as pd


def load_data(x):
    m = pd.read_csv(x, sep=',')
    m.head()
    return m


def into_dataframe(x):
    m = pd.DataFrame(x)
    m.head()
    return m


def get_features(x, y):
    m = x.drop(y, axis=1)
    return m


def get_labels(x, y):
    m = x.pop(y)
    m.head()
    return m


def data_preprocessing(loc, target, loc_specified='Yes'):
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    if loc_specified == 'Yes':
        data = load_data(loc)
        df0 = into_dataframe(data)
        df0_features = get_features(df0, target)
        df0_labels = get_labels(df0, target)
    else:
        df0_features = loc
        df0_labels = target
    # simple pre-processing
    dft_features = pd.DataFrame(StandardScaler().fit_transform(df0_features), columns=df0_features.columns)
    dft_labels = pd.DataFrame(LabelEncoder().fit_transform(df0_labels), columns=[target.name])
    x = dft_features
    y = dft_labels
    return x, y


def get_class(loc):
    import os
    data = load_data(loc)
    df0 = into_dataframe(data)
    label = df0.columns.tolist()
    df0_labels = label[-1]
    return df0_labels


# <-------------------------------- Generating Dataset for Testing the toolkit --------------------------------------->

def synthetic_data():
    import pandas as pd
    from sklearn.datasets import make_regression
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import os

    current_directory = os.getcwd()
    # define dataset
    x, y, ac = make_regression(n_samples=1000, n_features=20, n_informative=13, n_targets=1, shuffle=True, coef=True,
                               noise=150, random_state=42)
    df_x = None
    df_y = None
    actual_coef = None
    if isinstance(x, np.ndarray):
        df_x = pd.DataFrame(x)
        df_y = pd.DataFrame(y, columns=['Target'])
        actual_coef = pd.DataFrame(ac)

    col_names = []
    for i in range(len(df_x.columns.values)):
        col_names.append(f"Feature-{i}")

    df_x.columns = col_names
    synthetic_data = pd.concat([df_x, df_y], axis=1)
    synthetic_data['Target'] = pd.cut(synthetic_data['Target'], bins=3, labels=False)
    actual_coef = pd.DataFrame(MinMaxScaler().fit_transform(actual_coef), columns=['actual_coef'])
    synthetic_data.to_csv(os.path.join(current_directory, r"Data", f"synthetic_data.csv"), header=True)
    actual_coef.to_csv(os.path.join(current_directory, r"Data", f"synthetic_data_ac.csv"), header=True)
    print('Class:', synthetic_data['Target'].value_counts())


# synthetic_data()