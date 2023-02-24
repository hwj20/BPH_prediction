# 导入所需的库
import numpy as np
from scipy.stats import chi2
import pandas as pd

# Load the data into a pandas DataFrame
df = pd.read_csv('data.csv')

# Define the number of bins
num_bins = 5



# 定义一个函数来计算卡方值
def cal_chi2(df):
    # 计算每个区间的正负样本数
    df['pos'] = df['label'].sum() - df['cumsum']
    df['neg'] = df['total'] - df['pos']
    # 计算每个区间的正负样本比例
    pos_ratio = df['pos'] / df['pos'].sum()
    neg_ratio = df['neg'] / df['neg'].sum()
    # 计算每个区间的期望值和实际值之差的平方除以期望值之和
    chi2_value = ((pos_ratio - neg_ratio) ** 2 / (pos_ratio + neg_ratio)).sum()
    return chi2_value


# 定义一个函数来进行卡方分箱
def chi_merge(df, col, label, max_bins=10):
    # 对变量进行排序并计算累积样本数和标签数
    col_df = pd.DataFrame({col: df[col], label: df[label]})
    col_df.sort_values(by=col, inplace=True)
    col_df.reset_index(drop=True, inplace=True)
    col_df['cumsum'] = col_df[label].cumsum()

    # 初始化分箱结果为单个区间，并计算初始卡方值
    bins_df = pd.DataFrame({col: col_df[col], 'total': 1, 'label': col_df[label]})
    bins_df[col + '_bin'] = bins_df[col]

    group_names = sorted(bins_df[col].unique())

    chi2_value = cal_chi2(bins_df)

    # 循环合并相邻区间，直到满足最大分箱数或者卡方值小于阈值为止

    while len(group_names) > max_bins:
        min_index = []
        min_chi2 = chi2_value

        for i in range(len(group_names) - 1):
            temp_names = group_names.copy()
            temp_names[i] = temp_names[i + 1]
            temp_bin_df = bins_df.groupby(temp_names).agg({col: ['min', 'max'], 'total': 'count', 'label': 'sum'})
            temp_bin_df.columns = ['min', 'max', 'total', 'label']
            temp_chi2_value = cal_chi2(temp_bin_df)

            if temp_chi2_value < min_chi2:
                min_index = i
        # 如果没有找到合并后卡方值更小的区间，说明已经达到最优分箱，退出循环
        if min_index == []:
            break

        # 否则，合并卡方值最小的相邻区间，并更新分箱结果和卡方值
        group_names[min_index] = group_names[min_index + 1]
        group_names.pop(min_index + 1)

        bins_df[col + '_bin'] = pd.cut(bins_df[col], group_names, right=False)

        bins_df = bins_df.groupby(col + '_bin').agg({col: ['min', 'max'], 'total': 'count', 'label': 'sum'})
        bins_df.columns = ['min', 'max', 'total', 'label']

        chi2_value = min_chi2

    # 返回分箱结果
    return bins_df


# 调用cal_chi2函数和chi_merge函数
bins_df = chi_merge(df, 'x', 'y', max_bins=10)

# 查看分箱结果
print(bins_df)