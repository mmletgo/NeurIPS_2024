import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


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


class OutputMLP(nn.Module):

    def __init__(self):
        super(OutputMLP, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 283)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


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

        # 5. 空间维度的线性层：聚合空间维度
        self.space_linear = nn.Linear(32, 1)

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
        self.output_layer = OutputMLP()  # 输出 (batch_size, 283)

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
        x = self.space_linear(x).squeeze(
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


# 保存与加载模型函数
def save_best_model(model,
                    optimizer,
                    val_loss,
                    best_val_loss,
                    path='best_model.pth',
                    target_min=0,
                    target_max=1):
    if val_loss < best_val_loss:
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': val_loss,
                'target_min': target_min,
                'target_max': target_max
            }, path)
        print(f"New best model saved with val_loss: {val_loss:.16f}")
        return val_loss
    return best_val_loss


def load_best_model(model, optimizer, path='best_model.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint['best_val_loss']
    target_min = checkpoint['target_min']
    target_max = checkpoint['target_max']
    print("Best model loaded.")
    return best_val_loss, target_min, target_max


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
    del signal_FGS
    dataset = np.concatenate([FGS_column[:, :, np.newaxis, :], signal_AIRS],
                             axis=2)
    del signal_AIRS, FGS_column

    data_train_tensor = torch.tensor(dataset).float()
    del dataset
    data_min = data_train_tensor.min(dim=1, keepdim=True)[0]
    data_max = data_train_tensor.max(dim=1, keepdim=True)[0]
    data_train_normalized = (data_train_tensor - data_min) / (data_max -
                                                              data_min)
    del data_train_tensor
    data_train_reshaped = data_train_normalized.permute(0, 2, 1, 3)
    targets_tensor = torch.tensor(targets).float()
    target_min = targets_tensor.min()
    target_max = targets_tensor.max()
    targets_normalized = (targets_tensor - target_min) / (target_max -
                                                          target_min)

    # 数据集拆分
    num_planets = data_train_reshaped.size(0)
    train_size = int(0.8 * num_planets)
    val_size = num_planets - train_size

    train_data, val_data = random_split(
        list(zip(data_train_reshaped, targets_normalized)),
        [train_size, val_size])

    # 数据加载器
    batch_size = 16
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = EnhancedTimeWavelengthTransformer().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    try:
        best_val_loss, target_min, target_max = load_best_model(
            model, optimizer)
    except:
        print("未找到最佳模型，开始训练。")

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
            torch.cuda.empty_cache()  # 每次批次处理后清理显存
        train_loss /= len(train_loader)

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for valid_batch_x, valid_batch_y in val_loader:
                valid_batch_x, valid_batch_y = valid_batch_x.cuda(
                ), valid_batch_y.cuda()
                valid_output = model(valid_batch_x).squeeze(-1)
                val_loss += criterion(valid_output, valid_batch_y).item()
                torch.cuda.empty_cache()  # 每次批次处理后清理显存
        val_loss /= len(val_loader)

        print(
            f'Epoch [{epoch + 1}], Train Loss: {train_loss:.16f}, Val Loss: {val_loss:.16f}'
        )
        best_val_loss = save_best_model(model,
                                        optimizer,
                                        val_loss,
                                        best_val_loss,
                                        target_max=target_max,
                                        target_min=target_min)

    best_val_loss, target_min, target_max = load_best_model(model, optimizer)
    print("训练完成并加载最佳模型。")
    with torch.no_grad():
        model.eval()
        results = []

        for i in range(0, data_train_reshaped.size(0), batch_size):
            batch = data_train_reshaped[i:i + batch_size].cuda()
            output = model(batch).cpu().numpy()
            results.append(output)

            torch.cuda.empty_cache()  # 每次批次处理后清理显存

        # 合并所有批次结果
        all_predictions = torch.tensor(np.concatenate(results, axis=0))
        all_predictions = all_predictions * (target_max -
                                             target_min) + target_min
        all_predictions = all_predictions.numpy()

        # 保存预测结果
        np.save("predicted_targets.npy", all_predictions)
        print("预测结果已保存。")


if __name__ == "__main__":
    main()
