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

parser = argparse.ArgumentParser(description='PARSER')
parser.add_argument('--name', type=str, default='test', help='Name of paths')
args = parser.parse_args()

#%%
train_path = './feature/norm50/feature_873_labeled.parquet'
test_path = './feature/norm50/test_873_labeled.parquet'

name = args.name
if name == 'None':
    print('No name.')
# name = 'None'
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
sym_feat = ['sym', 'date', 'time']
drop_cols = sym_feat + label_feat

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

#%%
def fit():
    names = pd.read_csv('./feature_806.csv', header=None)
    names = names.values[:, 0]
    names = np.concatenate((np.array(drop_cols), names))
    train_data = pd.read_parquet(train_path)
    train_data = train_data[names]

    test_data = pd.read_parquet(test_path)
    test_data = test_data[names]

    train_data = sort_time(train_data)
    train_data = train_data.replace([np.inf, -np.inf, np.nan], 0)
    test_data = test_data.replace([np.inf, -np.inf, np.nan], 0)
    test_index = test_data[sym_feat]

    Y = train_data[label_feat]
    X = train_data.drop(['time'] + label_feat, axis=1)
    test_X = test_data.drop(['time'] + label_feat, axis=1)
    test_X = test_X[X.columns]
    # X = X.iloc[:1000, :]
    # test_X = test_X.iloc[:1000, :]
    # test_index = test_index.iloc[:1000, :]
    # Y = Y.iloc[:1000, :]
    print(X.shape)
    print(test_X.shape)
    print(Y.shape)
    del train_data, test_data
    gc.collect()

    params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'tree_method': 'gpu_hist',  # 使用 GPU 加速的 tree_method
    # 'gpu_id': 1,  # 指定使用的 GPU 设备 ID
    'max_depth': 30,
    'eta': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    }

    # n_fold = 5
    # ids_folds = {}

    models = {}
    result_df = []
    for i, label in enumerate(['label_10']):
        print('Begin training', label, '...')

        res = None

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=False)

        y_train = Y_train[label]
        y_val = Y_val[label]

        # 创建 XGBoost 的数据矩阵
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(test_X)

        # 训练模型
        model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtrain, 'train'), (dval, 'val')], \
                          early_stopping_rounds=100, verbose_eval=20)
        predicted = model.predict(dtest)
        res = pd.Series(np.argmax(predicted, axis=-1), name=label)


        models[label] = model
        result_df.append(res)
        del X_train, X_val, Y_train, Y_val
        gc.collect()

    result_df = pd.concat(result_df, axis=1)
    result_df = pd.concat([test_index, result_df], axis=1)

    result_df.to_csv(os.path.join(result_path, 'results_feature_873_labeled.csv'))

if __name__ == '__main__':
    fit()
