import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class SelfAttentionPooling(nn.Module):

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.attention_feature = nn.Linear(input_dim, 1)  # 处理特征维度

    def forward(self, x):
        weights = F.softmax(self.attention_feature(x),
                            dim=1)  # (batch_size, seq_len, 1)
        pooled = torch.sum(x * weights, dim=1)  # (batch_size, input_dim)
        return pooled


class SmallTransformer(nn.Module):

    def __init__(self,
                 d_model_wavelength=283,
                 nhead=8,
                 num_layers=4,
                 dropout=0.1):
        super(SmallTransformer, self).__init__()

        self.expand_wavelength = nn.Conv1d(in_channels=d_model_wavelength,
                                           out_channels=512,
                                           kernel_size=1)

        # 时间维度的吸收峰特征提取 Transformer
        self.time_peak_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,  # num_wavelengths
                nhead=nhead,
                dropout=dropout,
                batch_first=True),
            num_layers=num_layers)

        self.attention_pooling = SelfAttentionPooling(512)

        # 输出层：预测吸收峰值
        self.output_layer = MLP(512, 256, 283,
                                dropout=0.1)  # 输出 (batch_size, 283)

    def forward(self, x):
        batch_size, seq_len, num_wavelengths = x.size()
        x = x.permute(0, 2, 1)  # (batch_size, num_wavelengths, seq_len)
        x = self.expand_wavelength(x)  # (batch_size, 512, seq_len)
        x = x.permute(0, 2, 1)
        x = self.time_peak_transformer(x)  # (batch_size, seq_len, 512)
        x = self.attention_pooling(x)  # (batch_size, 512)
        # print(x.size())

        # 最终输出层：预测吸收峰值
        return self.output_layer(x)  # 输出 (batch_size, num_wavelengths)


if __name__ == "__main__":
    from utility import train_predict2
    train_predict2(SmallTransformer,
                   modelname="ST_v1.0",
                   batch_size=256,
                   train_epochs=1000)
