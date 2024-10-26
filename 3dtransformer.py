import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class MultiDimensionalTransformer(nn.Module):

    def __init__(self,
                 d_model_time=192,
                 d_model_wavelength=288,
                 nhead=4,
                 num_layers=2,
                 dropout=0.1):
        super(MultiDimensionalTransformer, self).__init__()

        # 1. MLP 升维：对同一个矩阵先进行时间升维，再进行波长升维
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


# 保存与加载模型函数
def save_best_model(model,
                    optimizer,
                    val_loss,
                    best_val_loss,
                    path='best_model_3d.pth',
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


def load_best_model(model, optimizer, path='best_model_3d.pth'):
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
    data_train_reshaped = data_train_normalized.permute(
        0, 2, 1, 3)  # (batch_size, num_wavelengths, seq_len, space_dim)
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
    model = MultiDimensionalTransformer().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    try:
        best_val_loss, target_min, target_max = load_best_model(
            model, optimizer)
    except:
        print("未找到最佳模型，开始训练。")

    # 训练循环
    for epoch in range(100):
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
        np.save("predicted_targets_3d.npy", all_predictions)
        print("预测结果已保存。")


if __name__ == "__main__":
    main()
