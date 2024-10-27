from torch import nn


class MultiDimensionalTransformer(nn.Module):

    def __init__(self,
                 d_model_time=192,
                 d_model_wavelength=288,
                 nhead=4,
                 num_layers=2,
                 dropout=0.1):
        super(MultiDimensionalTransformer, self).__init__()

        # 1. 卷积升维：对同一个矩阵先进行时间升维，再进行波长升维
        self.expand_time = nn.Conv1d(in_channels=187,
                                     out_channels=d_model_time,
                                     kernel_size=1)
        self.expand_wavelength = nn.Conv1d(in_channels=283,
                                           out_channels=d_model_wavelength,
                                           kernel_size=1)

        # 2. 定义 6 个 Transformer 层
        self.time_space_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)

        self.time_wavelength_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model_wavelength,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)

        self.space_time_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model_time,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)

        self.space_wavelength_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model_wavelength,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)

        self.wavelength_time_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model_time,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)

        self.wavelength_space_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)

        # 最后的 MLP 层：将输出映射为目标维度 (batch_size, 283)
        self.final_mlp = nn.Sequential(
            nn.Linear(d_model_time * d_model_wavelength * 32, 512), nn.ReLU(),
            nn.Linear(512, 283))

    def forward(self, x):
        batch_size, num_wavelengths, seq_len, space_dim = x.size()

        # 1. 对时间维度升维
        x = x.reshape(
            -1, seq_len,
            space_dim)  # (batch_size * num_wavelengths, seq_len, space_dim)
        x = self.expand_time(
            x)  # (batch_size * num_wavelengths, new_seq_len, space_dim)

        # 2. 对波长维度升维
        x = x.reshape(
            batch_size, num_wavelengths, -1, space_dim
        )  # (batch_size , num_wavelengths, new_seq_len, space_dim)
        batch_size, num_wavelengths, new_seq_len, space_dim = x.size()
        x = x.permute(
            0, 2, 1,
            3)  # (batch_size, new_seq_len, num_wavelengths, space_dim)
        x = x.reshape(
            -1, num_wavelengths, space_dim
        )  # (batch_size * new_seq_len, num_wavelengths, space_dim)
        x = self.expand_wavelength(
            x)  # (batch_size * new_seq_len, new_num_wavelengths, space_dim)
        x = x.reshape(
            batch_size, new_seq_len, -1, space_dim
        )  # (batch_size, new_seq_len, new_num_wavelengths, space_dim)
        batch_size, new_seq_len, new_num_wavelengths, space_dim = x.size()

        # 3. 顺序通过 6 个 Transformer 层
        x = x.permute(
            0, 2, 1,
            3)  # (batch_size, new_num_wavelengths, new_seq_len, space_dim)
        x = x.reshape(
            -1, new_seq_len, space_dim
        )  # (batch_size * new_num_wavelengths, new_seq_len, space_dim)
        x = self.time_space_transformer(x)  # 时间-空间

        x = x.reshape(
            batch_size, new_num_wavelengths, new_seq_len, space_dim
        )  # (batch_size, new_num_wavelengths, new_seq_len, space_dim)
        x = x.permute(
            0, 3, 2,
            1)  # (batch_size, space_dim, new_seq_len, new_num_wavelengths)
        x = x.reshape(
            -1, new_seq_len, new_num_wavelengths
        )  # (batch_size * space_dim, new_seq_len, new_num_wavelengths)
        x = self.time_wavelength_transformer(x)  # 时间-波长

        x = x.reshape(
            batch_size, space_dim, new_seq_len, new_num_wavelengths
        )  # (batch_size, space_dim, new_seq_len, new_num_wavelengths)
        x = x.permute(
            0, 3, 1,
            2)  # (batch_size, new_num_wavelengths, space_dim, new_seq_len)
        x = x.reshape(
            -1, space_dim, new_seq_len
        )  # (batch_size * new_num_wavelengths, space_dim, new_seq_len)
        x = self.space_time_transformer(x)  # 空间-时间

        x = x.reshape(
            batch_size, new_num_wavelengths, space_dim, new_seq_len
        )  # (batch_size, new_num_wavelengths, space_dim, new_seq_len)
        x = x.permute(
            0, 3, 2,
            1)  # (batch_size, new_seq_len, space_dim, new_num_wavelengths)
        x = x.reshape(-1, space_dim, new_num_wavelengths)
        x = self.space_wavelength_transformer(x)  # 空间-波长

        x = x.reshape(
            batch_size, new_seq_len, space_dim, new_num_wavelengths
        )  # (batch_size, new_seq_len, space_dim, new_num_wavelengths)
        x = x.permute(
            0, 2, 3,
            1)  # (batch_size, space_dim, new_num_wavelengths, new_seq_len)
        x = x.reshape(
            -1, new_num_wavelengths, new_seq_len
        )  # (batch_size * space_dim, new_num_wavelengths, new_seq_len)
        x = self.wavelength_time_transformer(x)  # 波长-时间

        x = x.reshape(
            batch_size, space_dim, new_num_wavelengths, new_seq_len
        )  # (batch_size, space_dim, new_num_wavelengths, new_seq_len)
        x = x.permute(
            0, 3, 2,
            1)  # (batch_size, new_seq_len, new_num_wavelengths, space_dim)
        x = x.reshape(
            -1, new_num_wavelengths, space_dim
        )  # (batch_size*new_seq_len, new_num_wavelengths, space_dim)
        x = self.wavelength_space_transformer(x)  # 波长-空间

        x = x.reshape(batch_size, new_seq_len, new_num_wavelengths, space_dim)
        # 4. 最终 MLP 输出层
        x = x.reshape(batch_size, -1)  # 展平为 (batch_size, feature_dim)
        output = self.final_mlp(x)  # 输出 (batch_size, 283)

        return output


if __name__ == "__main__":
    from utility import train_predict
    train_predict(MultiDimensionalTransformer,
                  modelname="3d_v1.0",
                  batch_size=16,
                  train_epchos=100)
