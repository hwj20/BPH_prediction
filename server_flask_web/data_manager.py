# encoding=utf-8
import os
import time
from datetime import datetime

import pandas as pd

model_path = '..//checkpoints'


def get_all_model(using_records=True):
    data = {'id': [], 'type': [], 'belongTo': [], 'updateDate': []}
    df = pd.DataFrame(data)
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M")
    for model_file in os.listdir(model_path):
        df = pd.concat([df, pd.DataFrame({'id': [model_file], 'type': ['机器学习'], 'belongTo': ['admin'],
                                          'updateDate': [current_time]})], sort=False)

    if using_records:
        return df.to_dict('records')
    else:
        return df.to_dict('list')


def upload_files():
    pass
