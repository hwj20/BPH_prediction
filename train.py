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
    feature_analysis('drink_state', read_file)
    input()
read_file = process_data(read_file)
selected_features = find_top_feature(read_file)
selected_features.append('is_BPH')
read_file = read_file[selected_features]
print(read_file.columns)
# input()


features = read_file
# Labels are the values we want to predict
labels = np.array(features['is_BPH'])
# Remove the labels from the features
features = features.drop('is_BPH', axis=1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                            random_state=42)

# train_mean(train_features, train_labels, test_features, test_labels)
train_all(train_features, train_labels, test_features, test_labels)
# train(train_features, train_labels, test_features, test_labels, method='')


"""
# with selected label
# method is RandomForest
# Mean Absolute Error: 0.07 degrees.
# Accuracy: 0.96
# AUC: 0.9589728453364817
# method is LogisticRegression
# Mean Absolute Error: 0.07 degrees.
# Accuracy: 0.93
# AUC: 0.9329988193624558
# method is DecisionTree
# Mean Absolute Error: 0.04 degrees.
# Accuracy: 0.96
# AUC: 0.9589728453364817
# method is GaussianNB
# Mean Absolute Error: 0.16 degrees.
# Accuracy: 0.84
# AUC: 0.859504132231405
# method is SVM
# Mean Absolute Error: 0.13 degrees.
# Accuracy: 0.87
# AUC: 0.871310507674144
# method is MLP
# Mean Absolute Error: 0.05 degrees.
# Accuracy: 0.95
# AUC: 0.9510035419126328
# Mean Absolute Error: 0.02 degrees.
# Accuracy: 0.98
# AUC: 0.9787485242030697
# method is XGBoost
# Mean Absolute Error: 0.25 degrees.
# Accuracy: 0.97
# AUC: 0.9663518299881936

# Mean Method
# Mean Absolute Error: 0.09294581092969938 degrees.
# f1score: 0.9636963696369636
# specificity: 0.9752066115702479
# Accuracy: 0.96
# AUC: 0.9616292798110979
# {'f1-score': 0.9636963696369636, 'precision': 0.9798657718120806, 'recall': 0.948051948051948, 'accuracy': 0.96, 'specificity': 0.9752066115702479}
"""
