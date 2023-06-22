# encoding=utf-8
import json
import math

import numpy
import pandas as pd
from pandas import DataFrame

from patient_type_utils import *

# 游离PSA和前列腺增生关系最大

unit_map = {}
data = []

read_file = pd.read_csv('../data/healthy_data_origin.csv', encoding='utf-8')

# 读取检验值
for index, row in read_file.iterrows():
    patient_id = row['ID_Patient']
    target_patient = PatientType(patient_id)
    target_patient.is_BPH = False

    for val in row.keys():
        if val in c2e_name_map.keys():
            if c2e_name_map[val] == 'PRO':
                if '阴性' in row[val]:
                    row[val] = 0
                elif row[val] != '':
                    PRO_val = row[val].split('(')[0]
                    PRO_val = float(PRO_val)
                    PRO_val = 1 if PRO_val > 0.15 else 0
                    row[val] = PRO_val
            target_patient.test_res[c2e_name_map[val]] = row[val]

    target_patient.age = row['Age']
    # target_patient
    # 处理其它疾病
    row_disease_names = ['现病史', '家族史', '既往史']
    target_patient.other_disease_info = '#'.join([row[name] for name in row_disease_names])

    has_other_disease = False
    for disease_info in target_patient.other_disease_info.split('#'):
        none_type = ['无', '', '-', '^', '─', '情况不明']
        # 所有 ‘无’, '', nan, '-', '^' 等作为没有疾病处理
        if type(disease_info) is not str:
            continue
        if disease_info not in none_type:
            has_other_disease = True
            continue
    target_patient.has_other_disease = has_other_disease

    # 处理手术史
    surgery_history = row['手术史']
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
    has_smoke_history = row['吸烟史']
    if type(has_smoke_history) == str and get_first_chinese_char(has_smoke_history) != '无':
        smoke_state = 2  # 抽烟
    if '戒烟' in has_smoke_history and smoke_state == 2:
        smoke_state = 1  # 戒烟
    target_patient.smoke_state = smoke_state

    # 处理饮酒史
    drink_state = 2     # 经常 和 其它标签
    has_drink_history = row['饮酒史']
    if type(has_drink_history) != str or get_first_chinese_char(has_drink_history) == '无':
        drink_state = 0         # 无饮酒史
    elif get_first_chinese_char(has_drink_history) == '有' or get_first_chinese_char(has_drink_history) == '偶':
        drink_state = 1         # 偶尔
    target_patient.drink_state = drink_state

    data.append(target_patient)


import mysql.connector
data_output = []

# Establish a connection to the MySQL database
cnx = mysql.connector.connect(
    host='localhost',
    user='root',
    password='123456',
    database='patient_data'
)

# Create a cursor object to execute SQL queries
cursor = cnx.cursor()
for patient in data:
    # Prepare the SQL statement
    sql = "INSERT INTO patient (patient_unique_number, age, is_BPH, other_info, test_res, has_other_disease, has_surgery, drink_state, smoke_state) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"

    # Prepare the values to be inserted
    values = (
        patient.patient_unique_number,
        patient.age,
        patient.is_BPH,
        json.dumps(patient.other_info),
        json.dumps(patient.test_res),
        patient.has_other_disease,
        patient.has_surgery,
        patient.drink_state,
        patient.smoke_state
    )

    # Execute the SQL statement
    cursor.execute(sql, values)

    # Commit the changes to the database
    cnx.commit()
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
df.to_csv('../data/healthy_data.csv', index=False, encoding='utf_8_sig')
cursor.close()
cnx.close()
print('insert database success!')

# translate the English label names into Chinese
# df = df.rename(columns=e2c_name_map)
# 创建新的转义字典
# rename_map = {}
# for key, val in e2c_name_map.items():
#     if val in unit_map.keys():
#         val = val + '(' + unit_map[val] + ')'
#     rename_map[key] = val
# df = df.rename(columns=rename_map)
