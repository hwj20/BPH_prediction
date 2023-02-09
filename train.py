import math
from ml_utils import train, train_all

import numpy
import pandas as pd
import numpy as np


#
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
        if val_list and len(val_list) != len(read_file):
            print(label)
            print(len(val_list))
            print('error!')
        if val_list:
            read_file[label] = val_list
    return read_file


read_file = pd.read_csv('data/train.csv', low_memory=False)

read_file = read_file.drop('GLU', 1)  # 没有
read_file = read_file.drop('ACR', 1)  # 没有
read_file = read_file.drop('登记号', 1)  # 多余信息
read_file = read_file.drop('病案号', 1)  # 多余信息
read_file = read_file.drop('就诊ID', 1) # 多余信息
read_file = read_file.drop('patient_unique_number', 1) # 多余信息
# 删去单位并填补空缺值
read_file = remove_units(read_file)
read_file = read_file.astype({col: np.int8 for col in read_file.columns[read_file.dtypes == np.bool_]})
# print(read_file.describe())
# count_label(read_file)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Use numpy to convert to arrays
import numpy as np

features = read_file
# Labels are the values we want to predict
labels = np.array(features['is_BPH'])
# Remove the labels from the features
# axis 1 refers to the columns
features = features.drop('is_BPH', axis=1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                            random_state=42)

train_all(train_features, train_labels, test_features, test_labels)
# train(train_features, train_labels, test_features, test_labels, method='')


# method is LogisticRegression
# Mean Absolute Error: 0.03 degrees.
# Accuracy: 0.97
# method is DecisionTree
# Mean Absolute Error: 0.04 degrees.
# Accuracy: 0.96
# method is GaussianNB
# Mean Absolute Error: 0.08 degrees.
# Accuracy: 0.92
# method is SVM
# Mean Absolute Error: 0.12 degrees.
# Accuracy: 0.88
# method is MLP
# Mean Absolute Error: 0.03 degrees.
# Accuracy: 0.97
# method is GBC
# Mean Absolute Error: 0.01 degrees.
# Accuracy: 0.99
