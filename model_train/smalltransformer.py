import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        # 定义线性层、激活层和 dropout
        # Define linear layers, activation, and dropout
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 前向传播过程
        # Forward pass
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class SelfAttentionPooling(nn.Module):

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        # 定义线性层，用于生成注意力权重
        # Define a linear layer for generating attention weights
        self.attention_feature = nn.Linear(input_dim, 1)

    def forward(self, x):
        # 计算注意力权重，并进行归一化
        # Compute attention weights and normalize
        weights = F.softmax(self.attention_feature(x),
                            dim=1)  # (batch_size, seq_len, 1)
        # 使用注意力权重进行加权平均
        # Weighted sum using attention weights
        pooled = torch.sum(x * weights, dim=1)  # (batch_size, input_dim)
        return pooled


class SmallTransformer(nn.Module):

    def __init__(self,
                 d_model_wavelength=283,
                 nhead=8,
                 num_layers=4,
                 dropout=0.1):
        super(SmallTransformer, self).__init__()

        # 吸收峰面积映射为嵌入向量
        # Map peak areas to embedding vector
        self.peak_embedding_fc = nn.Linear(284, 512)

        # 使用 1D 卷积扩展波长维度
        # Use 1D convolution to expand wavelength dimension
        self.expand_wavelength = nn.Conv1d(in_channels=d_model_wavelength,
                                           out_channels=512,
                                           kernel_size=1)

        # 时间维度的吸收峰特征提取 Transformer
        # Transformer for absorption peak feature extraction along time dimension
        self.time_peak_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,  # num_wavelengths
                nhead=nhead,
                dropout=dropout,
                batch_first=True),
            num_layers=num_layers)

        # 自注意力池化层
        # Self-attention pooling layer
        self.attention_pooling = SelfAttentionPooling(512)

        # 输出层：预测吸收峰值
        # Output layer: predict absorption peak values
        self.output_layer = MLP(512, 256, 283,
                                dropout=0.1)  # 输出 (batch_size, 283)
        # Output (batch_size, 283)

    def forward(self, x, peak_areas):
        # batch_size, seq_len, num_wavelengths = x.size()

        # 将吸收峰面积映射到嵌入向量空间
        # Map peak areas to embedding vector space
        peak_embed = self.peak_embedding_fc(peak_areas)  # (batch_size, 512)
        peak_embed = peak_embed.unsqueeze(1)  # (batch_size, 1, 512)

        # 转置输入以匹配卷积层
        # Transpose input to match Conv layer
        x = x.permute(0, 2, 1)  # (batch_size, num_wavelengths, seq_len)
        x = self.expand_wavelength(x)  # (batch_size, 512, seq_len)
        x = x.permute(0, 2, 1)

        # 将吸收峰面积嵌入与光通量数据融合
        # Fuse peak area embeddings with flux data
        x = x + peak_embed  # 广播加法
        # Broadcasting addition

        # 使用 Transformer 提取时间和吸收峰特征
        # Extract temporal and peak features using Transformer
        x = self.time_peak_transformer(x)  # (batch_size, seq_len, 512)
        x = self.attention_pooling(x)  # (batch_size, 512)

        # 通过输出层得到最终预测值
        # Obtain final predicted values through output layer
        return self.output_layer(x)  # 输出 (batch_size, num_wavelengths)
        # Output (batch_size, num_wavelengths)


if __name__ == "__main__":
    from utility import train_predict3
    # 训练模型
    # Train the model
    train_predict3(SmallTransformer,
                   modelname="ST_v3.3",
                   batch_size=512,
                   train_epochs=10000)
