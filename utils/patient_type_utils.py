# encoding=gbk
import re

# 中译英
c2e_name_map = {'红细胞计数': 'RBCs', '白细胞': 'WBC', '糖化血红蛋白A1c': 'GHb_A1c', '红细胞压积': 'PCV',
                # '红细胞': 'RBCs',
                '尿葡萄糖': 'GLU', '尿蛋白定量': 'PRO', '餐后2小时血糖': 'PBG', '空腹血糖': 'FBG', '尿蛋白定性':'PRO',
                '尿微量白蛋白': 'ACR',
                '游离PSA': 'FPSA', '甘油三酯': 'TAG', '低密度脂蛋白': 'LDL', '高密度脂蛋白': 'HDL', '胆固醇': 'CHOL',
                '葡萄糖': 'Glc', '尿酸': 'UA', '血小板计数': 'PLT'}

# 英译中
e2c_name_map = {"patient_unique_number": "患者唯一号", "age": "年龄", "is_BPH": "是否是 BPH 患者", "PCV": "红细胞压积",
                "WBC": "白细胞", "RBCs": "红细胞计数", "GHb_A1c": "糖化血红蛋白A1c", "GLU": "尿葡萄糖",
                "FPSA": "游离PSA",
                "TAG": "甘油三酯", "LDL": "低密度脂蛋白", "HDL": "高密度脂蛋白", "CHOL": "胆固醇", "Glc": "葡萄糖",
                "UA": "尿酸", "PLT": "血小板计数", "PRO": "尿蛋白定量", "PBG": "餐后2小时血糖", "FBG": "空腹血糖",
                "has_other_disease": "有确诊其它疾病", "has_surgery": "有手术史", "drink_state": "饮酒史",'ACT':'尿微量白蛋白',
                "smoke_state": "吸烟史"
                }
# 烟酒状态编码
w2n_drink_map = {'无': 0, '偶尔': 1, '经常': 2}
n2w_drink_map = {value: key for key, value in w2n_drink_map.items()}
w2n_smoke_map = {'无': 0, '戒烟': 1, '未戒烟': 2}
n2w_smoke_map = {value: key for key, value in w2n_smoke_map.items()}


def get_patient_by_number(patient_unique_number, patient_data: list):
    """
    find patient by patient_unique_number in patient_data
    :param patient_unique_number: the unique ID to recognize patient
    :param patient_data: the dictionary to be searched
    :return: the res to search the res in the dictionary patient_data
    """
    for patient in patient_data:
        if patient.patient_unique_number == patient_unique_number:
            return patient
    return None


def get_first_chinese_char(text):
    """
    find the first Chinese Character
    :param text: input text
    :return: first Chinese char
    """
    match = re.search(r'[\u4e00-\u9fa5]', text)  # This regular expression pattern match any unicode character from
    # '\u4e00' to '\u9fa5' which is the range for all chinese characters
    if match:
        return match.group()
    else:
        return None


class PatientType:
    def __init__(self, patient_num: int):
        self.patient_unique_number = patient_num  # 患者唯一号
        self.age = 0  # 年龄
        self.is_BPH = True  # 标签数据：是否有 BPH 这个病

        self.other_info = {}

        self.other_info['就诊ID'] = 0
        self.other_info['登记号'] = 0
        self.other_info['病案号'] = 0

        self.test_res = {}
        self.test_res['PCV'] = None  # 红细胞压积
        self.test_res['WBC'] = None  # 白细胞
        self.test_res['RBCs'] = None  # 红细胞
        self.test_res['GHb_A1c'] = None  # 糖化血红蛋白
        self.test_res['GLU'] = None  # 尿葡萄糖
        self.test_res['FPSA'] = None  # 游离PSA
        self.test_res['TAG'] = None  # 甘油三酯
        self.test_res['LDL'] = None  # 低密度脂蛋白
        self.test_res['HDL'] = None  # 高密度脂蛋白
        self.test_res['CHOL'] = None  # 胆固醇
        self.test_res['Glc'] = None  # 葡萄糖
        self.test_res['UA'] = None  # 尿酸
        self.test_res['PLT'] = None  # 血小板计数
        self.test_res['PRO'] = None  # 尿蛋白定量 （0.15 区分阴性阳性）
        self.test_res['ACR'] = None  # 尿微量白蛋白
        self.test_res['PBG'] = None  # 餐后2小时血糖
        self.test_res['FBG'] = None  # 空腹血糖`

        self.has_other_disease = False  # 被确诊其它疾病
        self.has_surgery = False  # 有手术史
        self.drink_state = 0  # 饮酒史，编码见上
        self.smoke_state = 0  # 吸烟史，编码见上

        self.other_disease_info = '未输入'
