# 超参数和路径配置

# ---- Data Settings ---- #
data_path = 'D:/a_aKaifa/Python_project/TMSA-Net/database/'  # 数据集路径
train_files = ['training.mat']            # 训练数据集文件名
test_files = ['evaluation.mat']           # 测试数据集文件名
output = 'output'                         # 保存输出（模型、日志等）的目录
model_name = "test"                       # 保存模型的名称
batch_size = 16                           # 训练和测试的批次大小
num_segs = 8                              # 数据增强的段数 沿着时间轴切成的段数

# ---- Model Settings ---- #
pool_size = 50                            # 池化核大小
pool_stride = 15                          # 池化步长
num_heads = 4                             # 注意力头数
fc_ratio = 2                              # 前馈网络扩展比例
depth = 1                                 # 变压器编码器深度（层数）
# ---- Training Settings ---- #
epochs = 2000                             # 训练轮数
lr = 2 ** -12                             # 学习率
weight_decay = 1e-4                       # 权重衰减