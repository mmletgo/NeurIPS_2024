import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# 定义增强版 Transformer 模型
class EnhancedTimeWavelengthTransformer(nn.Module):

    def __init__(self,
                 d_model_time=187,
                 d_model_wavelength=283,
                 nhead=4,
                 num_layers=2,
                 dropout=0.1):
        super(EnhancedTimeWavelengthTransformer, self).__init__()

        # 卷积层用于降噪预处理
        self.conv1d = nn.Conv1d(in_channels=32,
                                out_channels=32,
                                kernel_size=3,
                                padding=1)

        # 时间维度的降噪 Transformer
        self.time_denoising_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)

        # 波长维度的降噪 Transformer
        self.wavelength_denoising_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)

        # 线性层：用于压缩空间维度
        self.space_linear = nn.Linear(32, 1)

        # 时间维度的吸收峰特征提取 Transformer
        self.time_peak_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model_time,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)

        # 波长维度的吸收峰特征提取 Transformer
        self.wavelength_peak_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model_wavelength,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)

        # 时间维度的线性层：聚合时间信息
        self.time_linear = nn.Linear(d_model_time, 1)

        # 输出层：预测每个波长的吸收峰值
        self.output_layer = nn.Linear(d_model_wavelength, d_model_wavelength)

    def forward(self, x):
        batch_size, num_wavelengths, seq_len, space_dim = x.size()

        # 1. 卷积降噪预处理
        x = x.permute(0, 1, 3, 2).reshape(-1, space_dim, seq_len)
        x = self.conv1d(x).view(batch_size, num_wavelengths, -1, seq_len)

        # 2. 时间维度的降噪 Transformer
        x = x.permute(0, 1, 3, 2).reshape(-1, seq_len, space_dim)
        x = self.time_denoising_transformer(x)

        # 3. 波长维度的降噪 Transformer
        x = x.view(batch_size, num_wavelengths, seq_len,
                   -1).permute(0, 2, 1, 3)
        x = x.reshape(batch_size * seq_len, num_wavelengths, -1)
        x = self.wavelength_denoising_transformer(x)

        # 4. 线性层聚合空间维度
        x = x.view(batch_size, seq_len, num_wavelengths, -1)
        x = self.space_linear(x).squeeze(
            -1)  # (batch_size, seq_len, num_wavelengths)

        # 5. 吸收峰特征提取：时间维度
        x = self.time_peak_transformer(x)

        # 6. 吸收峰特征提取：波长维度
        x = x.permute(0, 2, 1)  # (batch_size, num_wavelengths, seq_len)
        x = self.wavelength_peak_transformer(x)

        # 7. 线性层聚合时间信息
        x = self.time_linear(x).squeeze(-1)  # (batch_size, 283)

        # 8. 输出层：预测吸收峰值
        return self.output_layer(x)  # 输出 (batch_size, 283)


# 模型保存与加载函数
def save_best_model(model,
                    optimizer,
                    val_loss,
                    best_val_loss,
                    path='best_model.pth'):
    if val_loss < best_val_loss:
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, path)
        print(f"New best model saved with val_loss: {val_loss:.5f}")
        return val_loss
    return best_val_loss


def load_best_model(model, optimizer, path='best_model.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Best model loaded.")


def main():
    # 数据加载与预处理
    auxiliary_folder = 'input/ariel-data-challenge-2024/'
    train_solution = np.loadtxt(f'{auxiliary_folder}/train_labels.csv',
                                delimiter=',',
                                skiprows=1)
    targets = train_solution[:, 1:]

    data_folder = 'input/binned-dataset-v3/'
    signal_AIRS = np.load(f'{data_folder}/data_train.npy')
    signal_FGS = np.load(f'{data_folder}/data_train_FGS.npy')

    FGS_column = signal_FGS.sum(axis=2)
    dataset = np.concatenate([FGS_column[:, :, np.newaxis, :], signal_AIRS],
                             axis=2)

    data_train_tensor = torch.tensor(dataset).float()
    data_min = data_train_tensor.min(dim=1, keepdim=True)[0]
    data_max = data_train_tensor.max(dim=1, keepdim=True)[0]
    data_train_normalized = (data_train_tensor - data_min) / (data_max -
                                                              data_min)
    data_train_reshaped = data_train_normalized.permute(0, 2, 1, 3)
    targets_tensor = torch.tensor(targets).float()

    # 数据集拆分
    num_planets = data_train_reshaped.size(0)
    train_size = int(0.8 * num_planets)
    val_size = num_planets - train_size

    train_data, val_data = random_split(
        list(zip(data_train_reshaped, targets_tensor)), [train_size, val_size])

    # 数据加载器
    batch_size = 16
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = EnhancedTimeWavelengthTransformer().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            optimizer.zero_grad()
            output = model(batch_x).squeeze(-1)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                output = model(batch_x).squeeze(-1)
                val_loss += criterion(output, batch_y).item()
        val_loss /= len(val_loader)

        print(
            f'Epoch [{epoch + 1}/50], Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}'
        )
        best_val_loss = save_best_model(model, optimizer, val_loss,
                                        best_val_loss)

    load_best_model(model, optimizer)


if __name__ == "__main__":
    main()
