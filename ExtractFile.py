# 用于在csv中提取特征向量
import pandas as pd
import pickle
import numpy as np

train_path = "Files/csvs/camel-1.4-o.csv"
test_path = "Files/csvs/camel-1.6-o.csv"

features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam',
            'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']

train_raw = pd.read_csv(train_path)
test_raw = pd.read_csv(test_path)

ascii()


def extract_feature(df, file_name):
    """
    提取手工标注的特征
    :param df:
    :param file_name:   csv的file_name字段对应
    :return:        [features]
    """
    row = df[df.file_name == file_name][features]
    row = np.array(row).tolist()
    row = np.squeeze(row)
    row = list(row)
    return row


# def extract_label(df, file_name):
#     """
#     提取这个文件是否有bug的标记
#     :param df:
#     :param file_name:
#     :return:
#     """
#     result = []
#     row = df[df.file_name == file_name]['bug']
#     row = np.array(row).tolist()
#     if row[0] >= 1:
#         result = [0, 1]
#     else:
#         result = [1, 0]
#     return result


def extract_label(df, file_name):
    """
    提取这个文件是否有bug的标记
    :param df:
    :param file_name:
    :return:
    """

    row = df[df.file_name == file_name]['bug']
    row = np.array(row).tolist()
    if row[0] > 1:
        row[0] = 1
    return row


def get_train_data():
    cnn_x_data = []
    hand_x_data = []
    label_data = []
    with open("Files/dump_data/camel-1.4/dict_ast_features", 'rb') as file_obj:
        ast = pickle.load(file_obj)
        for key, value in ast.items():
            cnn_x_data.append(value)
            hand_x_data.append(extract_feature(train_raw, key))
            label_data.append(extract_label(train_raw, key))
        cnn_x_data = np.array(cnn_x_data)
        hand_x_data = np.array(hand_x_data)
        label_data = np.array(label_data)
    return cnn_x_data, hand_x_data, label_data


def get_test_data():
    cnn_x_data = []
    hand_x_data = []
    label_data = []
    with open("Files/dump_data/camel-1.6/dict_ast_features", 'rb') as file_obj:
        ast = pickle.load(file_obj)
        for key, value in ast.items():
            cnn_x_data.append(value)
            hand_x_data.append(extract_feature(test_raw, key))
            label_data.append(extract_label(test_raw, key))
        cnn_x_data = np.array(cnn_x_data)
        hand_x_data = np.array(hand_x_data)
        label_data = np.array(label_data)
    return cnn_x_data, hand_x_data, label_data
