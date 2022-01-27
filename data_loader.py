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



