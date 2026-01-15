# # 在当前激活的py310TMSA环境中执行
# import torch
# print('PyTorch版本:', torch.__version__)
# print('CUDA是否可用:', torch.cuda.is_available())
# print('CUDA版本:', torch.version.cuda)
# print('显卡名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无GPU')

# import scipy.io as sio

# # 替换为你的文件路径
# mat_file_path = 'D:/a_aKaifa/Python_project/TMSA-Net/database/subject1/training.mat' 

# # 加载 .mat 文件
# data = sio.loadmat(mat_file_path)

# # 打印所有的键（变量名）
# print("Loaded .mat file keys (variable names):")
# print(data.keys())

import scipy.io as sio
import numpy as np

def load_data(file_path):
    # 加载.mat文件
    data = sio.loadmat(file_path)
    
    # ========== 新增：打印所有字段名 ==========
    print("=====.mat文件中的所有字段名=====")
    for key in data.keys():
        # 过滤掉MATLAB自动生成的默认字段（以__开头的）
        if not key.startswith('__'):
            print(f"字段名：{key}，数据形状：{data[key].shape}")
    # ========== 打印结束 ==========
    
    # 先注释掉报错的行，先运行看字段名
    # EEG = data['EEG_data'].astype(np.float64)
    # return EEG

load_data('D:/a_aKaifa/Python_project/TMSA-Net/database/subject1/training.mat')