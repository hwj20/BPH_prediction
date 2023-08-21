import math

from sklearn.model_selection import train_test_split

from utils.feature_select import *
from utils.ml_utils import train_all, train_mean

import numpy
import pandas as pd


#
# TODO 手术史 病史---数据清洗

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


analyse_feature = False
read_file = pd.read_csv('data/train.csv', low_memory=False)
if analyse_feature:
    feature_analysis('age', read_file)
    input()
read_file = process_data(read_file)
selected_features = find_top_feature(read_file)
selected_features.append('is_BPH')
read_file = read_file[selected_features]
print(read_file.columns)


features = read_file
# Labels are the values we want to predict
labels = np.array(features['is_BPH'])
# Remove the labels from the features
features = features.drop('is_BPH', axis=1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
# Split the data into training, validation and testing sets
train_features, X_temp, train_labels, y_temp = train_test_split(features, labels, test_size=0.4, random_state=42)
val_features, test_features, val_labels, test_labels = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
#                                                                             random_state=42)

print('-'*20+'validation'+'-'*20)
# train_mean(train_features, train_labels, val_features, val_labels)
train_all(train_features, train_labels, val_features, val_labels)
print('-'*20+'test'+'-'*20)
# train_mean(train_features, train_labels, test_features, test_labels)
train_all(train_features, train_labels, test_features, test_labels)


