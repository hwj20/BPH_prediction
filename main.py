import math
from ml_utils import train

import numpy
import pandas as pd


#
# 数据吐槽：餐后2小时血糖数据太少了；性别全为男性，这个数据有必要吗
# TODO 吸烟 手术史 病史 饮酒---数据清洗

def count_label(read_file):
    """
    print labels with counts
    :param read_file: read_csv
    :return: none
    """
    for label in read_file.columns.values.tolist():
        count_map = {}
        for val in read_file[label]:
            if isinstance(val, int) or isinstance(val, float) or ('0' <= val[0] <= '9'):
                if val == numpy.NaN:
                    print(val)
                continue
            if val not in count_map.keys():
                count_map[val] = 1
            else:
                count_map[val] += 1
        print(label)
        print(count_map)


def remove_units(read_file):
    """
    remove all the unit for measurement
    :param read_file: read_csv
    :return: read_file after removing unit for measurement
    """
    for label in read_file.columns.values.tolist():
        val_list = []
        for val in read_file[label]:
            if isinstance(val, int) or isinstance(val, float):
                if not math.isnan(val):
                    val_list.append(val)
                else:
                    val_list.append(0)  # 用 0 填补空缺数值
                    continue
            elif '0' <= val[0] <= '9':
                tmp = []
                for ctr in val:
                    if '0' <= ctr <= '9' or ctr == '.':
                        tmp.append(ctr)
                val_list.append(float(''.join(tmp)))

        i = 1
        if val_list and len(val_list) != 4542:
            print(label)
            print(len(val_list))
            print('error!')
        if val_list:
            read_file[label] = val_list
    return read_file


read_file = pd.read_csv('data/data_sample.csv', low_memory=False)

# 编一个 label, (label=1) 1824
actual = []
for res in read_file['彩超结果']:
    if '长大' in res:
        actual.append(1)
    else:
        actual.append(0)
read_file['actual'] = actual
read_file = read_file.drop('登记时间', 1)  # 全是同一时间，且与疾病无关
read_file = read_file.drop('性别', 1)  # 全是'男'
read_file = read_file.drop('餐后2小时血糖', 1)  # 数据太少了
# 删去文本数据 TODO：文本特征提取
read_file = read_file.drop('现病史', 1)
read_file = read_file.drop('手术史', 1)
read_file = read_file.drop('饮酒', 1)
read_file = read_file.drop('吸烟', 1)
read_file = read_file.drop('彩超结果', 1)
read_file = read_file.drop('彩超描述', 1)
# 删去
read_file = remove_units(read_file)
# print(read_file.describe())
# count_label(read_file)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Use numpy to convert to arrays
import numpy as np

features = read_file
# Labels are the values we want to predict
labels = np.array(features['actual'])
# Remove the labels from the features
# axis 1 refers to the columns
features = features.drop('actual', axis=1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                            random_state=42)

train(train_features, train_labels, test_features, test_labels)
