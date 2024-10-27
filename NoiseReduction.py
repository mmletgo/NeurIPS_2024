import numpy as np
from torch import nn


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


class EnhancedTimeWavelengthTransformer(nn.Module):

    def __init__(self,
                 d_model_time=187,
                 d_model_wavelength=283,
                 nhead=4,
                 num_layers=2,
                 dropout=0.1):
        super(EnhancedTimeWavelengthTransformer, self).__init__()

        # 1. 卷积层用于降噪预处理
        self.conv1d = nn.Conv1d(in_channels=32,
                                out_channels=32,
                                kernel_size=3,
                                padding=1)

        # 2. 时间维度和波长维度的升维
        self.expand_time = nn.Conv1d(in_channels=d_model_time,
                                     out_channels=256,
                                     kernel_size=1)
        self.expand_wavelength = nn.Conv1d(in_channels=d_model_wavelength,
                                           out_channels=512,
                                           kernel_size=1)

        # 3. 时间维度的降噪 Transformer
        self.time_denoising_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)

        # 4. 波长维度的降噪 Transformer
        self.wavelength_denoising_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)

        # 5. 空间维度的mlp层：聚合空间维度
        self.space_mlp = MLP(32, 16, 1, dropout=0.1)

        # 6. 时间维度的吸收峰特征提取 Transformer
        self.time_peak_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,  # num_wavelengths
                nhead=nhead,
                dropout=dropout,
                batch_first=True),
            num_layers=num_layers)

        # 7. 波长维度的吸收峰特征提取 Transformer
        self.wavelength_peak_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,  # seq_len
                nhead=nhead,
                dropout=dropout,
                batch_first=True),
            num_layers=num_layers)

        # 8. 时间维度的MLP层：聚合时间信息
        self.time_mlp = MLP(256, 128, 1, dropout=0.1)

        # 9. 输出层：预测吸收峰值
        self.output_layer = MLP(512, 256, 283,
                                dropout=0.1)  # 输出 (batch_size, 283)

    def forward(self, x):
        batch_size, num_wavelengths, seq_len, space_dim = x.size()

        # 1. 卷积降噪
        x = x.permute(0, 1, 3, 2).reshape(
            -1, space_dim,
            seq_len)  # (batch_size * num_wavelengths, space_dim, seq_len)
        x = self.conv1d(x).view(
            batch_size, num_wavelengths, -1,
            seq_len)  # (batch_size, num_wavelengths, space_dim, seq_len)

        # 2. 时间维度的降噪 Transformer
        x = x.permute(0, 1, 3, 2).reshape(
            -1, seq_len,
            space_dim)  # (batch_size * num_wavelengths, seq_len, space_dim)
        x = self.time_denoising_transformer(
            x)  # (batch_size * num_wavelengths, seq_len, space_dim)

        # 3. 波长维度的降噪 Transformer
        x = x.view(batch_size, num_wavelengths, seq_len,
                   -1)  # (batch_size, num_wavelengths, seq_len, space_dim)
        x = x.permute(0, 2, 1,
                      3)  # (batch_size, seq_len, num_wavelengths, space_dim)
        x = x.reshape(batch_size * seq_len, num_wavelengths,
                      -1)  # (batch_size * seq_len, num_wavelengths, space_dim)
        x = self.wavelength_denoising_transformer(
            x)  # (batch_size * seq_len, num_wavelengths, space_dim)

        # 4. 空间维度线性聚合
        x = x.view(batch_size, seq_len, num_wavelengths,
                   -1)  # (batch_size, seq_len, num_wavelengths, space_dim)
        x = self.space_mlp(x).squeeze(
            -1)  # (batch_size, seq_len, num_wavelengths)

        # 升维
        x = self.expand_time(x)  # (batch_size, new_seq_len, num_wavelengths)

        x = x.permute(0, 2, 1)  # (batch_size, num_wavelengths,new_seq_len)
        x = self.expand_wavelength(
            x)  # (batch_size, new_num_wavelengths,new_seq_len)

        # 5. 时间维度吸收峰特征提取
        x = x.permute(0, 2,
                      1)  # (batch_size, new_seq_len, new_num_wavelengths)
        x = self.time_peak_transformer(
            x)  # (batch_size , new_seq_len, new_num_wavelengths)

        # 6. 波长维度吸收峰特征提取
        x = x.permute(0, 2,
                      1)  # (batch_size, new_num_wavelengths, new_seq_len)
        x = self.wavelength_peak_transformer(
            x)  # (batch_size, new_num_wavelengths, new_seq_len)

        # 7. 时间维度的聚合
        x = self.time_mlp(x).squeeze(-1)

        # 8. 最终输出层：预测吸收峰值
        return self.output_layer(x)  # 输出 (batch_size, num_wavelengths)


if __name__ == "__main__":
    from utility import train_predict
    train_predict(EnhancedTimeWavelengthTransformer,
                  modelname="NR_v2.0",
                  batch_size=16,
                  train_epchos=100)
