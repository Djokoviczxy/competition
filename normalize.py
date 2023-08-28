import pandas as pd
import numpy as np
import os
import pickle
import time
from tqdm import tqdm
import gc
# from sklearn.model_selection import StratifiedKFold   # 补
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import f1_score
import argparse
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

train_path = './feature/feature_873_labeled.parquet'
test_path = './feature/test_873_labeled.parquet'

label_feat = ['label_5','label_10','label_20', 'label_40', 'label_60']
drop_cols = ['sym', 'date', 'time'] + label_feat

def norm(x):
    tmp = x.copy()
    tmp = tmp.drop(columns=drop_cols, axis=1)
    rolling_mean = tmp.rolling(window=50, min_periods=30).mean()
    rolling_std = tmp.rolling(window=50, min_periods=30).std() + 1e-9
    tmp = tmp.sub(rolling_mean).div(rolling_std)
    # 将所有 float64 列转换为 float32
    tmp = tmp.astype({col: 'float32' for col in tmp.select_dtypes(include='float64').columns})
    tmp[drop_cols] = x[drop_cols]
    return tmp

def handler(data):

    am_data = data[data['time'] < '12:00:00']
    pm_data = data[data['time'] > '12:00:00']

    am_data = norm(am_data)
    pm_data = norm(pm_data)
    res = pd.concat([am_data, pm_data], axis=0)
    return res

def normalize(data):
    data = data.groupby(['sym', 'date']).apply(handler).reset_index(drop=True)
    return data

train_data = pd.read_parquet(train_path)
train_data = normalize(train_data)
test_data = pd.read_parquet(test_path)
test_data = normalize(test_data)
train_data.to_parquet('feature/feature_873_labeled_norm50.parquet')
test_data.to_parquet('feature/test_873_labeled_norm50.parquet')