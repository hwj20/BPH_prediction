# encoding=gbk
import math

import numpy
import pandas as pd
from pandas import DataFrame

from PatientTypeUtils import *

# ����PSA��ǰ����������ϵ���

unit_map = {}
data = []

read_file = pd.read_csv('data/test_res.csv', low_memory=False, encoding='gbk')
# ��ȡ����ֵ
for index, row in read_file.iterrows():
    unit_name = row['��λ']
    label_name = row['��������Ŀ��������']
    patient_number = row['����Ψһ��']
    # �������ֵȱʧ��pass ����һ��
    if math.isnan(row['����ֵ']):
        continue
    # ��ͳ�ƺ�ϸ����һ��
    if label_name == '��ϸ��':
        continue
    # ɾ����ϸ����'/HP'��λ��ֵ
    if label_name == '��ϸ��' and unit_name == '/HP':
        continue
    # ɾ�������'umol/L'����ĵ�λ��ֵ
    if label_name == '����' and unit_name != 'umol/L':
        continue

    target_patient = get_patient_by_number(patient_number, data)
    if target_patient is None:
        new_patient = PatientType(patient_number)
        data.append(new_patient)
        target_patient = data[-1]

    english_label_name = c2e_name_map[label_name]  # ��Ӣ����д�洢�� PatientType.test_res ��
    target_patient.test_res[english_label_name] = row['����ֵ']
    target_patient.other_info['����ID'] = row['����ID']
    target_patient.other_info['�ǼǺ�'] = row['�ǼǺ�']
    target_patient.other_info['������1'] = row['������']
    target_patient.other_info['������2'] = row['������']

    if unit_name != '':
        unit_map[label_name] = unit_name

read_file = pd.read_csv('data/base_info.csv', low_memory=False, encoding='gbk')
# ��ȡ����
for index, row in read_file.iterrows():
    age = row['����']
    patient_number = row['����Ψһ��']
    # �������ȱʧ��pass ����һ�Ȼ����û�У�
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
# ��ȡ���漲��
for index, row in read_file.iterrows():
    patient_number = row['����Ψһ��']

    # �ñ��� 116 ��û�� patient_number�����û����
    if math.isnan(patient_number):
        continue

    other_disease = row['Ժ���������']

    none_type = ['��', '', '-', '^', '��', '�������']
    # ���� ���ޡ�, '', nan, '-', '^' ����Ϊû�м�������
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
# ��ȡ���漲��
for index, row in read_file.iterrows():
    patient_number = row['����Ψһ��']
    if math.isnan(patient_number):
        continue
    target_patient = get_patient_by_number(patient_number, data)

    # ȷ�����߳����ڱ���
    if target_patient is None:
        continue
        # new_patient = PatientType(patient_number)
        # data.append(new_patient)
        # target_patient = data[-1]

    # ��������ʷ
    surgery_history = row['����ʷ-����ʷ']
    has_surgery_history = False

    # ���� ���ޡ�, '', nan, '-', '^' �Ȳ�ָ��
    if type(surgery_history) is not str:
        pass
    if get_first_chinese_char(surgery_history) == '��' or get_first_chinese_char(surgery_history) == '��':
        pass
    else:
        has_surgery_history = True

    if not target_patient.has_surgery and has_surgery_history:
        target_patient.has_surgery = True
        target_patient.surgery_history = surgery_history

    # ��������ʷ
    smoke_state = 0     # �޳���ʷ
    has_smoke_history = row['����ʷ-����']
    quit_smoke_history = row['����ʷ-����']
    if type(has_smoke_history) == str and get_first_chinese_char(has_smoke_history) != '��':
        smoke_state = 2  # ����
    if type(quit_smoke_history) == str and smoke_state == 2 \
            and (get_first_chinese_char(quit_smoke_history) == '��'
                 or get_first_chinese_char(quit_smoke_history) == '��'):
        smoke_state = 1  # ����
    target_patient.smoke_state = smoke_state

    # ��������ʷ
    drink_state = 2     # ���� �� ������ǩ
    has_drink_history = row['����ʷ-�Ⱦ�']
    if type(has_drink_history) != str or get_first_chinese_char(has_drink_history) == '��':
        drink_state = 0         # ������ʷ
    elif get_first_chinese_char(has_drink_history) == '��' or get_first_chinese_char(has_drink_history) == 'ż':
        drink_state = 1         # ż��
    target_patient.drink_state = drink_state

    # �������ʷ TODO

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
# �����µ�ת���ֵ�
rename_map = {}
for key, val in e2c_name_map.items():
    if val in unit_map.keys():
        val = val + '(' + unit_map[val] + ')'
    rename_map[key] = val
df = df.rename(columns=rename_map)
df.to_csv('patient_data.csv', index=False, encoding='gbk')

# ������
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
