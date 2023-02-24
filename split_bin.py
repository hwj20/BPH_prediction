# ��������Ŀ�
import numpy as np
from scipy.stats import chi2
import pandas as pd

# Load the data into a pandas DataFrame
df = pd.read_csv('data.csv')

# Define the number of bins
num_bins = 5



# ����һ�����������㿨��ֵ
def cal_chi2(df):
    # ����ÿ�����������������
    df['pos'] = df['label'].sum() - df['cumsum']
    df['neg'] = df['total'] - df['pos']
    # ����ÿ�������������������
    pos_ratio = df['pos'] / df['pos'].sum()
    neg_ratio = df['neg'] / df['neg'].sum()
    # ����ÿ�����������ֵ��ʵ��ֵ֮���ƽ����������ֵ֮��
    chi2_value = ((pos_ratio - neg_ratio) ** 2 / (pos_ratio + neg_ratio)).sum()
    return chi2_value


# ����һ�����������п�������
def chi_merge(df, col, label, max_bins=10):
    # �Ա����������򲢼����ۻ��������ͱ�ǩ��
    col_df = pd.DataFrame({col: df[col], label: df[label]})
    col_df.sort_values(by=col, inplace=True)
    col_df.reset_index(drop=True, inplace=True)
    col_df['cumsum'] = col_df[label].cumsum()

    # ��ʼ��������Ϊ�������䣬�������ʼ����ֵ
    bins_df = pd.DataFrame({col: col_df[col], 'total': 1, 'label': col_df[label]})
    bins_df[col + '_bin'] = bins_df[col]

    group_names = sorted(bins_df[col].unique())

    chi2_value = cal_chi2(bins_df)

    # ѭ���ϲ��������䣬ֱ�����������������߿���ֵС����ֵΪֹ

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
        # ���û���ҵ��ϲ��󿨷�ֵ��С�����䣬˵���Ѿ��ﵽ���ŷ��䣬�˳�ѭ��
        if min_index == []:
            break

        # ���򣬺ϲ�����ֵ��С���������䣬�����·������Ϳ���ֵ
        group_names[min_index] = group_names[min_index + 1]
        group_names.pop(min_index + 1)

        bins_df[col + '_bin'] = pd.cut(bins_df[col], group_names, right=False)

        bins_df = bins_df.groupby(col + '_bin').agg({col: ['min', 'max'], 'total': 'count', 'label': 'sum'})
        bins_df.columns = ['min', 'max', 'total', 'label']

        chi2_value = min_chi2

    # ���ط�����
    return bins_df


# ����cal_chi2������chi_merge����
bins_df = chi_merge(df, 'x', 'y', max_bins=10)

# �鿴������
print(bins_df)