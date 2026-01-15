import os 
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

import config # 超参数和路径配置
from eeg_dataset import seed_torch, load_data, eegDataset
from train import train_evaluation


# 构建学习率调度器 学习率随训练迭代按余弦曲线周期性下降
def build_lr_scheduler(optimizer, n_iter_per_epoch):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter_per_epoch * config.epochs)
    return scheduler

# 从路径中拿 训练数据和测试数据
def build_datasets_files(stage='train'):
    datasets = []  # 用于存储每个受试者的数据集文件路径的列表
    #                    ['training.mat']                    ['evaluation.mat']
    target_file = config.train_files if stage == 'train' else config.test_files
    for dir in sorted(os.listdir(config.data_path)):    # D:遍历dataset下的每一个文件
        if '.' in dir: 
            continue
        data_files = []
        for file in sorted(os.listdir(config.data_path + dir)): # 遍历每一个 subject 下的文件
            if file in target_file:  # 如果文件在目标文件列表中
                data_files.append(config.data_path + dir + '/' + file)  # 拼接成完整路径 
        if data_files:
            datasets.append(data_files)
    return datasets

def main():
    # 使用显卡训练
    torch.cuda.set_device(0)
    randomSeed = random.randint(1, 10000)
    print(f'seed is {randomSeed}')
    seed_torch(randomSeed)  # 设置随机种子
    print(f'device {0} is used for training')
    accuracy = []  # 存储每个被试的准确率
    kappa = []  # 存储每个被试的kappa
    
    # 训练数据集和测试数据集
    train_datasets = build_datasets_files(stage='train')
    test_datasets = build_datasets_files(stage='test')

    #遍历每个被试 
    for i in range(len(train_datasets)): 
        subject = train_datasets[i][0].split('/')[-2]  #倒数第二个文件夹名  subject1 subject2
        print(f'------start {subject} training------')

        # 创建 保存模型的路径
        save_path = os.path.join(config.output, 'models', config.model_name, subject)
        if not os.path.exists(save_path): #如果路径不存在则创建
            os.makedirs(save_path) 

        train_file = train_datasets[i]  #就是上面的darafiles 一个darafiles里面有一个training.mat
        test_file = test_datasets[i]

        #-------------  假设每个受试者只有1个数据文件
        train_data, train_labels = load_data(train_file[0]) #得到训练数据和标签
        test_data, test_labels = load_data(test_file[0])  #得到测试数据和标签

        # 创建数据集和数据加载器 变成tensor类型 放到显卡上
        train_dataset = eegDataset(torch.from_numpy(train_data).cuda(), torch.from_numpy(train_labels).long().cuda())
        test_dataset = eegDataset(torch.from_numpy(test_data).cuda(), torch.from_numpy(test_labels).long().cuda())

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True) # 16
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        # 初始化模型
        from network.TMSANet import TMSANet
        # ---- Model Initialization Tips ---- #
        # - Always set `radix = 1`.
        # - If using the BCIC-IV-2b dataset:
        #   Initialize the model as: TMSANet(3, 1, 1000, 2)
        #   where:
        #     - 3: Number of input channels
        #     - 1: Radix value
        #     - 1000: Number of time points
        #     - 2: Number of classes
        # - If using the HGD dataset:
        #   Initialize the model as: TMSANet(44, 1, 1125, 4)
        #   where:
        #     - 44: Number of input channels
        #     - 1: Radix value
        #     - 1125: Number of time points
        #     - 4: Number of classes
        model = TMSANet(22, 1, 1000, 4) #通道 radix 参数 时间点 分类数
        print('\n', model)

        # 将模型移动到GPU
        model.cuda()

        # 计算模型的可训练参数数量
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        # 设置损失函数、优化器和学习率调度器
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        scheduler = build_lr_scheduler(optimizer, len(train_loader))

        # 训练和评估模型
        best_acc, best_kappa = train_evaluation(model, train_loader, test_loader, criterion, optimizer, scheduler, save_path)
        accuracy.append(best_acc)
        kappa.append(best_kappa)

    # 打印每一个被试的准确率和kappa
    for i in range(len(accuracy)):
        print(f'subject:A0{i+1},accuracy:{accuracy[i]:.6f},kappa:{kappa[i]:.6f}')
    
    # 打印平均准确率和kappa
    print('average accuracy:', sum(accuracy) / len(accuracy), 'average kappa:', sum(kappa) / len(kappa), 'seed:', randomSeed)

if __name__ == '__main__':
    main()
