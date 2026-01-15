from scipy import io  # 读取.mat 文件
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# 用于封装 EEG 数据和标签
class eegDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.labels = label  # 存储标签
        self.data = data  # 存储数据

    def __getitem__(self, index): 
        #取出一个数据和标签  索引index
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        # 返回数据集的大小 多少个样本
        return len(self.data)

# 设置随机种子
def seed_torch(seed):
    random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁用哈希随机化
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)  # 设置 PyTorch GPU 随机种子
    torch.backends.cudnn.benchmark = False  # 禁用 CUDNN benchmark 以保证可复现性
    torch.backends.cudnn.deterministic = True  # 确保 CUDNN 的确定性行为

# 从.mat 文件中加载 EEG 数据
def load_data(data_file, stage='test'):
    print(f'加载文件: {data_file}')
    data = io.loadmat(data_file)

    # 标签编号必须从0开始
    # 数据的形状为 [1000,22,288] (T,C, N)       标签形状为 [288, 1] 
    EEG = data['data'].astype(np.float64) # 转为双精度浮点数
    # EEG = data['data_set'].astype(np.float64)  # 2b data tag
    # EEG = data['data'].astype(np.float64) # HGD data tag

    # 改变数据的形状到 [288, 22, 1000]   (N, C, T)
    EEG = EEG.transpose((2,1, 0))  # 使用 HGD 数据集, 请不要使用此行

    # 改变标签的形状到 [288,]
    labels = data['label'].reshape(-1).astype(np.int32)  # 把标签变成一维数组 32位整数
    # labels = data['labels_set'].reshape(-1).astype(np.int32)  # 2b labels tag
    # labels = data['labels'].reshape(-1).astype(np.int32) # HGD labels tag

    # 标签编号必须从0开始
    labels = labels - np.min(labels)

    # 打乱数据和标签的顺序
    EEG, labels = shuffle_data(EEG, labels)

    print(f'预处理的数据: {EEG.shape} {labels.shape}')
    return EEG, labels

# 功能是随机打乱数据和标签
def shuffle_data(data, label):
    index = [i for i in range(len(data))]  # 创建索引列表
    random.shuffle(index)  # 打乱索引列表
    shuffle_data = data[index]  # 重新排列数据
    shuffle_label = label[index]  # 重新排列标签
    return shuffle_data, shuffle_label
    #不能只打乱数据而不打乱标签，否则数据和标签就对不上了