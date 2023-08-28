import pandas as pd
import numpy as np
import os
import _pickle as pickle
import time
from tqdm import tqdm
import gc
# from sklearn.model_selection import StratifiedKFold   # 补
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from sklearn.metrics import f1_score
import argparse


parser = argparse.ArgumentParser(description='PARSER')
parser.add_argument('--name', type=str, default='None', help='Name of paths')
args = parser.parse_args()


train_path = 'feature/filtered_final/feature_873_labeled_norm50.parquet'
test_path = 'feature/filtered_final/test_873_labeled_norm50.parquet'

name = args.name
if name == 'None':
    print('No name.')





model_path = f'model/{name}/'            # 需要创建名为{model_path}的文件夹
if not os.path.isdir(model_path):
    os.makedirs(model_path)

id_path = f'id/{name}/'            # 需要创建名为{id_path}的文件夹
if not os.path.isdir(id_path):
    os.makedirs(id_path)

importance_path = f'importance/{name}/'            # 需要创建名为{importance_path}的文件夹
if not os.path.isdir(importance_path):
    os.makedirs(importance_path)

result_path = f'results/{name}/'            # 需要创建名为{result_path}的文件夹
if not os.path.isdir(result_path):
    os.makedirs(result_path)

label_feat = ['label_5','label_10','label_20', 'label_40', 'label_60']

def saveDict(data, path):
    pickle.dump(data, open(path, 'wb'), protocol=4)

def fromDict(path):
    print('loading file:', path, end=' ')
    t = time.time()
    data = pickle.loads(open(path,'rb').read())
    print('Time spent:', time.time() - t)
    return data


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = y_hat.reshape(-1,3)  # 3 分类
    y_hat = np.argmax(y_hat, axis=-1)
    return 'f1', f1_score(y_true, y_hat, average='macro'), True


def sort_time(X):
    levels = ['date', 'time', 'sym']
    return X.sort_values(by=levels)


def fit():
    train_data = pd.read_parquet(train_path)
    test_data = pd.read_parquet(test_path)
    train_data = sort_time(train_data)

    Y = train_data[label_feat]
    X = train_data.drop(['time'] + label_feat, axis=1)
    test_X = test_data.drop(['time'] + label_feat, axis=1)
    test_X = test_X[X.columns]
    print(X)
    print(test_X)

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        # 'metric': 'custom',  # 设置为'custom'，因为我们使用自定义的评估函数
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        # 'device': 'gpu',
        # 'gpu_platform_id': 0,
        # 'gpu_devices_id': 0
    }

    # n_fold = 5
    # ids_folds = {}
    models = {}
    result_df = []
    for i, label in enumerate(label_feat):
        print('Begin training', label, '...')

        # kf = KFold(n_splits=n_fold)
        res = None

        # for fold, (idx_tr, idx_va) in enumerate(kf.split(X)):
        callbacks = [log_evaluation(period=20), early_stopping(stopping_rounds=100)]

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=False)

        y_train = Y_train[label]
        y_val = Y_val[label]

        # 创建LightGBM数据集
        train_dataset = lgb.Dataset(X_train, label=y_train)
        val_dataset = lgb.Dataset(X_val, label=y_val)
        del X_train, X_val, y_train, y_val, Y_train, Y_val
        gc.collect()

        # 训练模型
        model = lgb.train(params, train_dataset, valid_sets=[train_dataset, val_dataset], num_boost_round=10000, callbacks=callbacks, feval=lgb_f1_score)
        predicted = model.predict(test_X)
        res = pd.Series(np.argmax(predicted, axis=-1), name=label)

        print('res')
        print(res)
        models[label] = model
        result_df.append(res)
 
    result_df = pd.concat(result_df, axis=1)
    result_df = pd.concat([test_data[['date', 'time', 'sym']], result_df], axis=1)

    result_df.to_csv(os.path.join(result_path, 'results_feature_873_labeled.csv'))

if __name__ == '__main__':
    fit()
