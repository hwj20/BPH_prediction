# encoding=gbk
import math

import numpy
import pandas as pd
from pandas import DataFrame

from PatientTypeUtils import *

# 游离PSA和前列腺增生关系最大

unit_map = {}
data = []

read_file = pd.read_csv('data/test_res.csv', low_memory=False, encoding='gbk')
# 读取检验值
for index, row in read_file.iterrows():
    unit_name = row['单位']
    label_name = row['检验子项目中文名称']
    patient_number = row['患者唯一号']
    # 如果检验值缺失，pass 掉这一项
    if math.isnan(row['检验值']):
        continue
    # 不统计红细胞这一栏
    if label_name == '红细胞':
        continue
    # 删除白细胞的'/HP'单位数值
    if label_name == '白细胞' and unit_name == '/HP':
        continue
    # 删除尿酸的'umol/L'以外的单位数值
    if label_name == '尿酸' and unit_name != 'umol/L':
        continue

    target_patient = get_patient_by_number(patient_number, data)
    if target_patient is None:
        new_patient = PatientType(patient_number)
        data.append(new_patient)
        target_patient = data[-1]

    english_label_name = c2e_name_map[label_name]  # 以英文缩写存储在 PatientType.test_res 中
    target_patient.test_res[english_label_name] = row['检验值']
    target_patient.other_info['就诊ID'] = row['就诊ID']
    target_patient.other_info['登记号'] = row['登记号']
    target_patient.other_info['病案号1'] = row['病案号']
    target_patient.other_info['病案号2'] = row['病案号']

    if unit_name != '':
        unit_map[label_name] = unit_name

read_file = pd.read_csv('data/base_info.csv', low_memory=False, encoding='gbk')
# 读取年龄
for index, row in read_file.iterrows():
    age = row['年龄']
    patient_number = row['患者唯一号']
    # 如果年龄缺失，pass 掉这一项（然而并没有）
    if math.isnan(age):
        continue

    target_patient = get_patient_by_number(patient_number, data)
    if target_patient is None:
        continue
        # new_patient = PatientType(patient_number)
        # data.append(new_patient)
        # target_patient = data[-1]

    target_patient.age = age

read_file = pd.read_csv('data/other_disease.csv', low_memory=False, encoding='gbk')
# 读取伴随疾病
for index, row in read_file.iterrows():
    patient_number = row['患者唯一号']

    # 该表有 116 行没有 patient_number，因而没处理
    if math.isnan(patient_number):
        continue

    other_disease = row['院内诊断名称']

    none_type = ['无', '', '-', '^', '─', '情况不明']
    # 所有 ‘无’, '', nan, '-', '^' 等作为没有疾病处理
    if type(other_disease) is not str:
        continue
    if other_disease in none_type:
        continue

    target_patient = get_patient_by_number(patient_number, data)
    if target_patient is None:
        continue
        # new_patient = PatientType(patient_number)
        # data.append(new_patient)
        # target_patient = data[-1]

    if not target_patient.has_other_disease:
        target_patient.has_other_disease = True
        target_patient.other_disease_name = other_disease

read_file = pd.read_csv('data/history_text.csv', low_memory=False, encoding='gb18030')
# 读取伴随疾病
for index, row in read_file.iterrows():
    patient_number = row['患者唯一号']
    if math.isnan(patient_number):
        continue
    target_patient = get_patient_by_number(patient_number, data)

    # 确定患者出现在表中
    if target_patient is None:
        continue
        # new_patient = PatientType(patient_number)
        # data.append(new_patient)
        # target_patient = data[-1]

    # 处理手术史
    surgery_history = row['既往史-手术史']
    has_surgery_history = False

    # 所有 ‘无’, '', nan, '-', '^' 等不指名
    if type(surgery_history) is not str:
        pass
    if get_first_chinese_char(surgery_history) == '无' or get_first_chinese_char(surgery_history) == '否':
        pass
    else:
        has_surgery_history = True

    if not target_patient.has_surgery and has_surgery_history:
        target_patient.has_surgery = True
        target_patient.surgery_history = surgery_history

    # 处理吸烟史
    smoke_state = 0     # 无抽烟史
    has_smoke_history = row['个人史-嗜烟']
    quit_smoke_history = row['个人史-戒烟']
    if type(has_smoke_history) == str and get_first_chinese_char(has_smoke_history) != '无':
        smoke_state = 2  # 抽烟
    if type(quit_smoke_history) == str and smoke_state == 2 \
            and (get_first_chinese_char(quit_smoke_history) == '戒'
                 or get_first_chinese_char(quit_smoke_history) == '已'):
        smoke_state = 1  # 戒烟
    target_patient.smoke_state = smoke_state

    # 处理饮酒史
    drink_state = 2     # 经常 和 其它标签
    has_drink_history = row['个人史-嗜酒']
    if type(has_drink_history) != str or get_first_chinese_char(has_drink_history) == '无':
        drink_state = 0         # 无饮酒史
    elif get_first_chinese_char(has_drink_history) == '有' or get_first_chinese_char(has_drink_history) == '偶':
        drink_state = 1         # 偶尔
    target_patient.drink_state = drink_state

    # 处理既往史 TODO

data_output = []
for patient in data:
    data_output.append({
        "patient_unique_number": patient.patient_unique_number,
        **patient.other_info,
        "age": patient.age,
        "is_BPH": patient.is_BPH,
        **patient.test_res,
        "has_other_disease": patient.has_other_disease,
        "has_surgery": patient.has_surgery,
        "drink_state": patient.drink_state,
        "smoke_state": patient.smoke_state
    })
df = pd.DataFrame(data_output)
# df = df.rename(columns=e2c_name_map)
# 创建新的转义字典
rename_map = {}
for key, val in e2c_name_map.items():
    if val in unit_map.keys():
        val = val + '(' + unit_map[val] + ')'
    rename_map[key] = val
df = df.rename(columns=rename_map)
df.to_csv('patient_data.csv', index=False, encoding='gbk')

# 输出结果
# for target_patient in data:
# print(target_patient.patient_unique_number)
#
# if target_patient.age == 0:
#     print('missing age!')
# print(target_patient.age)
#
# print('has other disease:' + target_patient.has_other_disease.__str__())
# if target_patient.has_other_disease:
#     print(target_patient.other_disease_name)
#
# print('has surgery:' + target_patient.has_surgery.__str__())
# if target_patient.has_surgery:
#     print(target_patient.surgery_history)
#
# print(target_patient.test_res.__str__())
