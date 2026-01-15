import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import config
# 多尺度的一维卷积模块
class MultiScaleConv1d(nn.Module):
    """
    多尺度一维卷积模块，用于提取具有多种卷积核大小的特征。
    Args: 
        in_channels: 输入通道数。
        out_channels: 每个卷积的输出通道数。
        kernel_sizes: 每个卷积层的卷积核大小列表。
        padding: 每个卷积核大小对应的填充列表。
    """
    def __init__(self, in_channels, out_channels, kernel_sizes, padding):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=p) for k, p in zip(kernel_sizes, padding)
        ])
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))  # 连接后进行批量归一化
        self.dropout = nn.Dropout(0.5)  # 正则化的 dropout

    def forward(self, x):
        # 对每个卷积应用并沿通道维度连接结果
        conv_outs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)  # 沿通道轴连接
        out = self.bn(out)  # 应用批量归一化
        out = self.dropout(out)  # 应用 dropout
        return out


# 多头注意力模块，结合局部和全局注意力
class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制，结合局部和全局注意力。
    参数:
        d_model: 输入特征的维度。
        n_head: 注意力头的数量。
        dropout: 正则化的 dropout 率。
    """
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = d_model // n_head  # 每个注意力头的键的维度
        self.d_v = d_model // n_head  # 每个注意力头的值的维度
        self.n_head = n_head

        # 多尺度卷积设置，用于局部特征提取
        kernel_sizes = [3, 5]
        padding = [1, 2]

        self.multi_scale_conv_k = MultiScaleConv1d(d_model, d_model, kernel_sizes, padding)

        # Queries, local keys, global keys, and values 的线性投影
        self.w_q = nn.Linear(d_model, n_head * self.d_k)
        self.w_k_local = nn.Linear(d_model * len(kernel_sizes), n_head * self.d_k)
        self.w_k_global = nn.Linear(d_model, n_head * self.d_k)
        self.w_v = nn.Linear(d_model, n_head * self.d_v)
        self.w_o = nn.Linear(n_head * self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        局部和全局注意力组合的前向传播。
        参数:
            query: Query 张量，形状为 (batch_size, seq_len, d_model)。
            key: Key 张量，形状为 (batch_size, seq_len, d_model)。
            value: Value 张量，形状为 (batch_size, seq_len, d_model)。
        """
        bsz = query.size(0)

        # 使用多尺度卷积提取局部键
        key_local = key.transpose(1, 2)  # 转置为 (batch_size, d_model, seq_len)
        key_local = self.multi_scale_conv_k(key_local).transpose(1, 2)

        # 线性投影
        q = self.w_q(query).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)  # Query
        k_local = self.w_k_local(key_local).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)  # Local Key
        k_global = self.w_k_global(key).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)  # Global Key
        v = self.w_v(value).view(bsz, -1, self.n_head, self.d_v).transpose(1, 2)  # Value

        # Local attention
        scores_local = torch.matmul(q, k_local.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_local = F.softmax(scores_local, dim=-1)
        attn_local = self.dropout(attn_local)
        x_local = torch.matmul(attn_local, v)

        # Global attention
        scores_global = torch.matmul(q, k_global.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_global = F.softmax(scores_global, dim=-1)
        attn_global = self.dropout(attn_global)
        x_global = torch.matmul(attn_global, v)

        # Combine local and global attention outputs
        x = x_local + x_global

        # Concatenate results and project to output dimensions
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.n_head * self.d_v)
        return self.w_o(x)


# 前馈神经网络 提供非线性能力
class FeedForward(nn.Module):
    """
    两层前馈神经网络，使用 GELU 激活函数。
    参数:
        d_model: 输入和输出特征的维度。
        d_hidden: 隐藏层的维度。
        dropout: 正则化的 dropout 率。
    """
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()  # 激活函数
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)  # 线性层 1
        x = self.act(x)  # 激活函数
        x = self.dropout(x)  # Dropout
        x = self.w_2(x)  # 线性层 2
        x = self.dropout(x)  # Dropout
        return x


# Transformer Encoder Layer 翻译: 
class TransformerEncoder(nn.Module):
    """
    一个包含多头注意力和前馈网络的单层 Transformer 编码器。
    参数:
        embed_dim: 输入嵌入的维度。
        num_heads: 注意力头的数量。
        fc_ratio: 前馈隐藏层扩展的比例。
        attn_drop: 注意力机制的 dropout 率。
        fc_drop: 前馈网络的 dropout 率。
    """
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.feed_forward = FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)  # LayerNorm after attention
        self.layernorm2 = nn.LayerNorm(embed_dim)  # LayerNorm after feed-forward

    def forward(self, data):
        # Apply attention with residual connection and layer normalization
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)

        # Apply feed-forward network with residual connection and layer normalization
        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output

# 特征提取模块
class ExtractFeature(nn.Module):
    """
    翻译: 使用卷积层从输入数据中提取时序和空间特征。
    参数:
        num_channels: 输入通道数（例如传感器或特征数量）。
        num_samples: 输入序列中的时间点数量。
        embed_dim: 嵌入的输出维度。
        pool_size: 平均池化的核大小。
        pool_stride: 平均池化的步幅大小。
    """
    def __init__(self, num_channels, num_samples, embed_dim, pool_size, pool_stride):
        super().__init__()
        # 不同卷积核大小的时序卷积
        self.temp_conv1 = nn.Conv2d(1, embed_dim, (1, 31), padding=(0, 15))
        self.temp_conv2 = nn.Conv2d(1, embed_dim, (1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(embed_dim)  # 时序特征的批归一化

        # 跨所有通道的空间卷积
        self.spatial_conv1 = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1), padding=(0, 0))
        self.bn2 = nn.BatchNorm2d(embed_dim)  # 空间特征的批归一化
        self.glu = nn.GELU()  # 激活函数
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)  # 时序平均池化
    def forward(self, x):
        """
        翻译: 特征提取的前向传播。
        参数:
            x: 形状为 (batch_size, num_channels, num_samples) 的输入张量。
        返回:
            提取特征后的输出张量。
        """
        x = x.unsqueeze(dim=1)  # 添加通道维度 -> (batch_size, 1, num_channels, num_samples)
        x1 = self.temp_conv1(x)  # 时序卷积，卷积核大小为31
        x2 = self.temp_conv2(x)  # 时序卷积，卷积核大小为15
        x = x1 + x2  # 结合两种卷积的特征
        x = self.bn1(x)  # 应用批归一化
        x = self.spatial_conv1(x)  # 空间卷积
        x = self.glu(x)  # 激活函数
        x = self.bn2(x)  # 应用批归一化
        x = x.squeeze(dim=2)  # 移除空间维度 -> (batch_size, embed_dim, num_samples)
        x = self.avg_pool(x)  # 应用平均池化
        return x


# Transformer Module
class TransformerModule(nn.Module):
    """
    堆叠多个 Transformer 编码器层。
    参数:
        embed_dim: 输入嵌入的维度。
        num_heads: 每个编码器层中的注意力头数量。
        fc_ratio: 前馈层的扩展比例。
        depth: Transformer 编码器层的数量。
        attn_drop: 注意力机制的 dropout 率。
        fc_drop: 前馈网络的 dropout 率。
    """
    def __init__(self, embed_dim, num_heads, fc_ratio, depth, attn_drop, fc_drop):
        super().__init__()
        # 创建多个 Transformer 编码器层的列表
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for _ in range(depth)
        ])

    def forward(self, x):
        """
        Transformer 模块的前向传播。
        参数:
            x: 形状为 (batch_size, embed_dim, num_samples) 的输入张量。
        返回:
            形状相同的变换后张量。
        """
        x = rearrange(x, 'b d n -> b n d')  # 重新排列为 (batch_size, seq_len, embed_dim)
        for encoder in self.transformer_encoders:
            x = encoder(x)  # 通过每个编码器层
        x = x.transpose(1, 2)  # 重新排列回 (batch_size, embed_dim, seq_len)
        x = x.unsqueeze(dim=2)  # 添加空间维度 -> (batch_size, embed_dim, 1, seq_len)
        return x


# Classification Module
class ClassifyModule(nn.Module):
    """
    基于提取的特征进行分类。
    参数:
        embed_dim: 嵌入的维度。
        temp_embedding_dim: 池化后时间嵌入的维度。
        num_classes: 输出类别的数量。
    """
    def __init__(self, embed_dim, temp_embedding_dim, num_classes):
        super().__init__()
        # 全连接层用于分类
        self.classify = nn.Linear(embed_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        """
        分类的前向传播。
        参数:
            x: 形状为 (batch_size, embed_dim, 1, seq_len) 的输入张量。
        返回:
            形状为 (batch_size, num_classes) 的分类 logits。
        """
        x = x.reshape(x.size(0), -1)  # 将输入张量展平
        out = self.classify(x)  # 通过分类层
        return out


# Complete TMSA-Net Model
class TMSANet(nn.Module):
    """
    TMSA-Net：结合特征提取、Transformer编码器和分类模块。
    参数:
        in_planes (int): 输入通道数（例如传感器数量）。
        radix (int): 基数因子，通常设置为1。
        time_points (int): 输入序列中的时间点数量。
        num_classes (int): 分类的输出类别数量。
        embed_dim (int): 嵌入的维度。
            - 对于 BCIC-IV-2a 使用 19。
            - 对于 BCIC-IV-2b 使用 6。
            - 对于 HGD 使用 10。
        pool_size (int): 池化的核大小。
        pool_stride (int): 池化的步幅大小。
        num_heads (int): Transformer 中注意力头的数量。
        fc_ratio (int): 前馈层的扩展比例。
        depth (int): Transformer 编码器的深度（层数）。
        attn_drop (float): 注意力机制的 dropout 率。
            - 对于 HGD 数据集设置为 0.7。
        fc_drop (float): 前馈层的 dropout 率。
    """
    def __init__(self, in_planes, radix, time_points, num_classes, embed_dim=19, pool_size=config.pool_size,
                 pool_stride=config.pool_stride, num_heads=config.num_heads, fc_ratio=config.fc_ratio, depth=config.depth, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.in_planes = in_planes * radix  # 调整输入维度
        self.extract_feature = ExtractFeature(self.in_planes, time_points, embed_dim, pool_size, pool_stride)
        temp_embedding_dim = (time_points - pool_size) // pool_stride + 1  # 计算时间嵌入大小
        self.dropout = nn.Dropout()  # Transformer 前的 dropout 层
        self.transformer_module = TransformerModule(embed_dim, num_heads, fc_ratio, depth, attn_drop, fc_drop)
        self.classify_module = ClassifyModule(embed_dim, temp_embedding_dim, num_classes)

    def forward(self, x):
        """
        TMSA-Net 的前向传播。
        参数:
            x: 形状为 (batch_size, in_planes, time_points) 的输入张量。
        返回:
            形状为 (batch_size, num_classes) 的分类 logits。
        """
        x = self.extract_feature(x)  # 提取特征
        x = self.dropout(x)  # 应用 dropout
        x = self.transformer_module(x)  # 应用 transformer 模块
        out = self.classify_module(x)  # 分类特征
        return out


# Main function to test the model
if __name__ == '__main__':
    # Instantiate the model
    block = TMSANet(22, 1, 1000, 4)

    # Generate random input data (batch_size=16, channels=22, time_points=1000)
    input = torch.rand(16, 22, 1000)

    # Perform forward pass
    output = block(input)

    # Calculate total number of trainable parameters
    total_trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')  # Print the total parameters

    # Print model architecture
    print(block)

