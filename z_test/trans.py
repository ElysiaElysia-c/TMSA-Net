import os
import numpy as np
import pyedflib
from scipy.io import savemat

def gdf2mat_bci4_2a(gdf_file_path, output_mat_path):
    """
    将BCI4-2a的.gdf文件转换为.mat文件
    适配模型需要的data/labels字段格式
    """
    # 1. 读取GDF文件
    f = pyedflib.EdfReader(gdf_file_path)
    
    # 2. 提取核心信息（适配BCI4-2a数据集）
    # 通道数（BCI4-2a是22个EEG通道）
    n_channels = f.signals_in_file
    # 采样频率（BCI4-2a默认250Hz）
    sfreq = f.getSampleFrequency(0)
    # 提取EEG数据（22通道 × 总采样点）
    eeg_data = []
    for ch in range(n_channels):
        eeg_data.append(f.readSignal(ch))
    eeg_data = np.array(eeg_data).T  # 转置为：总采样点 × 22通道
    print(f"原始EEG数据形状：{eeg_data.shape} (采样点 × 通道数)")
    
    # 3. 提取事件标签（BCI4-2a的标签对应：769=左手,770=右手,771=双脚,772=舌头）
    annotations = f.readAnnotations()
    labels = []
    label_times = []
    
    # 解析标注信息（时间戳、持续时间、标签）
    for i in range(len(annotations[0])):
        onset = annotations[0][i]  # 事件开始时间（秒）
        duration = annotations[1][i]
        label = annotations[2][i]
        
        # 只保留运动想象的标签（过滤其他无关标注）
        if label in ['769', '770', '771', '772']:
            # 转换为数字标签（1-4，方便模型训练）
            num_label = int(label) - 768
            labels.append(num_label)
            # 计算事件对应的采样点位置
            label_sample = int(onset * sfreq)
            label_times.append(label_sample)
    
    # 4. 截取试次数据（BCI4-2a每个试次通常是2s预备+4s想象，取想象阶段数据）
    # 可根据模型需求调整截取范围（比如取4s数据：1000个采样点，250Hz×4）
    trial_duration = 4  # 每个试次的时长（秒）
    trial_samples = int(trial_duration * sfreq)  # 4×250=1000个采样点
    trial_data = []
    trial_labels = []
    
    for i, start_sample in enumerate(label_times):
        end_sample = start_sample + trial_samples
        # 确保不超出数据范围
        if end_sample <= eeg_data.shape[0]:
            trial = eeg_data[start_sample:end_sample, :]
            trial_data.append(trial)
            trial_labels.append(labels[i])
    
    # 转换为numpy数组
    trial_data = np.array(trial_data)  # 形状：试次数 × 采样点 × 通道数
    trial_labels = np.array(trial_labels)  # 形状：试次数
    print(f"处理后试次数据形状：{trial_data.shape} (试次数 × 采样点 × 通道数)")
    print(f"处理后标签形状：{trial_labels.shape} (试次数)")
    
    # 5. 保存为.mat文件（匹配模型需要的data/labels字段）
    mat_data = {
        'data': trial_data.astype(np.float64),  # 模型读取的data字段
        'labels': trial_labels.astype(np.int32)  # 模型读取的labels字段
    }
    savemat(output_mat_path, mat_data)
    print(f"✅ 转换完成！.mat文件已保存到：{output_mat_path}")
    
    # 关闭文件
    f._close()
    return trial_data, trial_labels

def batch_convert_gdf2mat(input_dir, output_dir):
    """
    批量转换文件夹下的所有.gdf文件
    """
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历所有.gdf文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.gdf'):
            # 构建输入输出路径
            gdf_path = os.path.join(input_dir, file_name)
            mat_file_name = file_name.replace('.gdf', '.mat')
            mat_path = os.path.join(output_dir, mat_file_name)
            
            # 转换单个文件
            print(f"\n正在转换：{file_name}")
            try:
                gdf2mat_bci4_2a(gdf_path, mat_path)
            except Exception as e:
                print(f"❌ 转换{file_name}失败：{str(e)}")

if __name__ == "__main__":
    # ==================== 请修改这里的路径 ====================
    # GDF文件所在文件夹（比如BCI4-2a的subject1文件夹）
    input_dir = r"C:/Users/23996/Desktop/111"
    # 转换后的MAT文件保存文件夹
    output_dir = r"C:/Users/23996/Desktop/222"
    # =========================================================
    
    # 批量转换
    batch_convert_gdf2mat(input_dir, output_dir)
    
    # 如需转换单个文件，取消下面注释：
    # gdf_file = r"C:/Users/23996/Desktop/111/A01T.gdf"
    # mat_file = r"C:/Users/23996/Desktop/222/training.mat"
    # gdf2mat_bci4_2a(gdf_file, mat_file)