import os
import pandas as pd
import matplotlib.pyplot as plt
import random

data_path = '../data/xunfei/test/'
file_names = os.listdir(data_path)


def middle_price_calculate(data):
    bid1 = data['n_bid1']
    ask1 = data['n_ask1']
    if bid1 != 0 and ask1 != 0:
        middle_price = (bid1 + ask1) / 2
    elif bid1 != 0:
        middle_price = bid1
    elif ask1 != 0:
        middle_price = ask1
    else:
        middle_price = None
        raise ValueError('There are some problems')
    return middle_price


def direction_calculate(price_t0, price_future, door):
    price_range = price_future - price_t0
    if price_range < - door:
        return 0
    elif -door <= price_range <= door:
        return 1
    elif price_range > door:
        return 2


for file_name in file_names:
    direction_5_list = []
    direction_10_list = []
    direction_20_list = []
    direction_40_list = []
    direction_60_list = []
    index_list = []
    data = pd.read_csv(data_path + file_name)

    for i in range(len(data.index)):
        index_list.append(i)

    for i in range(len(data.index) - 5):
        t_0 = data.iloc[i]
        t_5 = data.iloc[i + 5]

        middle_price_0 = middle_price_calculate(t_0)
        middle_price_5 = middle_price_calculate(t_5)
        direction_5 = direction_calculate(middle_price_0, middle_price_5, 0.0005)
        direction_5_list.append(direction_5)

    for i in range(len(data.index) - 10):
        t_0 = data.iloc[i]
        t_10 = data.iloc[i + 10]

        middle_price_0 = middle_price_calculate(t_0)
        middle_price_10 = middle_price_calculate(t_10)
        direction_10 = direction_calculate(middle_price_0, middle_price_10, 0.0005)
        direction_10_list.append(direction_10)

    for i in range(len(data.index) - 20):
        t_0 = data.iloc[i]
        t_20 = data.iloc[i + 20]

        middle_price_0 = middle_price_calculate(t_0)
        middle_price_20 = middle_price_calculate(t_20)
        direction_20 = direction_calculate(middle_price_0, middle_price_20, 0.001)
        direction_20_list.append(direction_20)

    for i in range(len(data.index) - 40):
        t_0 = data.iloc[i]
        t_40 = data.iloc[i + 40]

        middle_price_0 = middle_price_calculate(t_0)
        middle_price_40 = middle_price_calculate(t_40)
        direction_40 = direction_calculate(middle_price_0, middle_price_40, 0.001)
        direction_40_list.append(direction_40)

    for i in range(len(data.index) - 60):
        t_0 = data.iloc[i]
        t_60 = data.iloc[i + 60]

        middle_price_0 = middle_price_calculate(t_0)
        middle_price_60 = middle_price_calculate(t_60)
        direction_60 = direction_calculate(middle_price_0, middle_price_60, 0.001)
        direction_60_list.append(direction_60)

    for i in range(5):
        direction_5_list.append(2)
    for i in range(10):
        direction_10_list.append(2)
    for i in range(20):
        direction_20_list.append(2)
    for i in range(40):
        direction_40_list.append(2)
    for i in range(60):
        direction_60_list.append(2)

    result = pd.DataFrame({
        'uuid': index_list,
        'label_5': direction_5_list,
        'label_10': direction_10_list,
        'label_20': direction_20_list,
        'label_40': direction_40_list,
        'label_60': direction_60_list
    })
    result.to_csv('./submit/' + file_name, index=None)
