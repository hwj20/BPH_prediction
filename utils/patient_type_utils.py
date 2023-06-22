# encoding=gbk
import re

# ����Ӣ
c2e_name_map = {'��ϸ������': 'RBCs', '��ϸ��': 'WBC', '�ǻ�Ѫ�쵰��A1c': 'GHb_A1c', '��ϸ��ѹ��': 'PCV',
                # '��ϸ��': 'RBCs',
                '��������': 'GLU', '�򵰰׶���': 'PRO', '�ͺ�2СʱѪ��': 'PBG', '�ո�Ѫ��': 'FBG', '�򵰰׶���':'PRO',
                '��΢���׵���': 'ACR',
                '����PSA': 'FPSA', '��������': 'TAG', '���ܶ�֬����': 'LDL', '���ܶ�֬����': 'HDL', '���̴�': 'CHOL',
                '������': 'Glc', '����': 'UA', 'ѪС�����': 'PLT'}

# Ӣ����
e2c_name_map = {"patient_unique_number": "����Ψһ��", "age": "����", "is_BPH": "�Ƿ��� BPH ����", "PCV": "��ϸ��ѹ��",
                "WBC": "��ϸ��", "RBCs": "��ϸ������", "GHb_A1c": "�ǻ�Ѫ�쵰��A1c", "GLU": "��������",
                "FPSA": "����PSA",
                "TAG": "��������", "LDL": "���ܶ�֬����", "HDL": "���ܶ�֬����", "CHOL": "���̴�", "Glc": "������",
                "UA": "����", "PLT": "ѪС�����", "PRO": "�򵰰׶���", "PBG": "�ͺ�2СʱѪ��", "FBG": "�ո�Ѫ��",
                "has_other_disease": "��ȷ����������", "has_surgery": "������ʷ", "drink_state": "����ʷ",'ACT':'��΢���׵���',
                "smoke_state": "����ʷ"
                }
# �̾�״̬����
w2n_drink_map = {'��': 0, 'ż��': 1, '����': 2}
n2w_drink_map = {value: key for key, value in w2n_drink_map.items()}
w2n_smoke_map = {'��': 0, '����': 1, 'δ����': 2}
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
        self.patient_unique_number = patient_num  # ����Ψһ��
        self.age = 0  # ����
        self.is_BPH = True  # ��ǩ���ݣ��Ƿ��� BPH �����

        self.other_info = {}

        self.other_info['����ID'] = 0
        self.other_info['�ǼǺ�'] = 0
        self.other_info['������'] = 0

        self.test_res = {}
        self.test_res['PCV'] = None  # ��ϸ��ѹ��
        self.test_res['WBC'] = None  # ��ϸ��
        self.test_res['RBCs'] = None  # ��ϸ��
        self.test_res['GHb_A1c'] = None  # �ǻ�Ѫ�쵰��
        self.test_res['GLU'] = None  # ��������
        self.test_res['FPSA'] = None  # ����PSA
        self.test_res['TAG'] = None  # ��������
        self.test_res['LDL'] = None  # ���ܶ�֬����
        self.test_res['HDL'] = None  # ���ܶ�֬����
        self.test_res['CHOL'] = None  # ���̴�
        self.test_res['Glc'] = None  # ������
        self.test_res['UA'] = None  # ����
        self.test_res['PLT'] = None  # ѪС�����
        self.test_res['PRO'] = None  # �򵰰׶��� ��0.15 �����������ԣ�
        self.test_res['ACR'] = None  # ��΢���׵���
        self.test_res['PBG'] = None  # �ͺ�2СʱѪ��
        self.test_res['FBG'] = None  # �ո�Ѫ��`

        self.has_other_disease = False  # ��ȷ����������
        self.has_surgery = False  # ������ʷ
        self.drink_state = 0  # ����ʷ���������
        self.smoke_state = 0  # ����ʷ���������

        self.other_disease_info = 'δ����'
