import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# 定义增强版 Transformer 模型
class EnhancedTimeWavelengthTransformer(nn.Module):

    def __init__(self, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(EnhancedTimeWavelengthTransformer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=32,
                                out_channels=32,
                                kernel_size=3,
                                padding=1)
        self.time_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)
        self.wavelength_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model,
                                       nhead=nhead,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        batch_size, num_wavelengths, seq_len, space_dim = x.size()
        x = x.permute(0, 1, 3, 2).reshape(-1, space_dim, seq_len)
        x = self.conv1d(x).view(batch_size, num_wavelengths, -1, seq_len)
        x = x.permute(0, 1, 3, 2).reshape(-1, seq_len, space_dim)
        x = self.time_transformer(x)
        x = x.view(batch_size, num_wavelengths, seq_len,
                   -1).permute(0, 2, 1, 3)
        x = x.reshape(batch_size * seq_len, num_wavelengths, -1)
        x = self.wavelength_transformer(x)
        x = x.view(batch_size, seq_len, num_wavelengths,
                   -1).permute(0, 2, 1, 3)
        return self.output_layer(x).squeeze(-1)


# 模型保存函数，仅在验证集上损失最低时保存模型
def save_best_model(model,
                    optimizer,
                    val_loss,
                    best_val_loss,
                    path='best_model.pth'):
    if val_loss < best_val_loss:
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
        print(f"New best model saved with val_loss: {val_loss:.5f}")
        return val_loss  # 更新最佳损失
    return best_val_loss  # 否则返回原来的最佳损失


def load_best_model(model, optimizer, path='best_model.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Best model loaded.")


def main():
    # 数据加载和处理
    auxiliary_folder = 'input/ariel-data-challenge-2024/'
    train_solution = np.loadtxt(f'{auxiliary_folder}/train_labels.csv',
                                delimiter=',',
                                skiprows=1)
    targets = train_solution[:, 1:]  # (673, 283)

    data_folder = 'input/binned-dataset-v3/'
    signal_AIRS_diff_transposed_binned = np.load(
        f'{data_folder}/data_train.npy')
    signal_FGS_diff_transposed_binned = np.load(
        f'{data_folder}/data_train_FGS.npy')

    FGS_column = signal_FGS_diff_transposed_binned.sum(axis=2)
    del signal_FGS_diff_transposed_binned

    dataset = np.concatenate(
        [FGS_column[:, :, np.newaxis, :], signal_AIRS_diff_transposed_binned],
        axis=2)
    del signal_AIRS_diff_transposed_binned, FGS_column

    data_train_tensor = torch.tensor(dataset).float()
    data_min = data_train_tensor.min(dim=1, keepdim=True)[0]
    data_max = data_train_tensor.max(dim=1, keepdim=True)[0]
    data_train_normalized = (data_train_tensor - data_min) / (data_max -
                                                              data_min)
    del data_train_tensor, dataset

    data_train_reshaped = data_train_normalized.permute(0, 2, 1, 3)
    del data_train_normalized

    targets_tensor = torch.tensor(targets).float()

    # 按行星划分训练集和验证集（80%训练，20%验证）
    num_planets = data_train_reshaped.size(0)
    train_size = int(0.8 * num_planets)
    val_size = num_planets - train_size

    train_data, val_data = random_split(
        list(zip(data_train_reshaped, targets_tensor)), [train_size, val_size])

    # 创建数据加载器
    batch_size = 16
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

    # 初始化模型、损失函数和优化器
    model = EnhancedTimeWavelengthTransformer().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')  # 初始化最佳验证损失为正无穷大

    # 训练循环
    num_epochs = 50
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            optimizer.zero_grad()

            output = model(batch_x).squeeze(-1)  # 输出 (batch_size, 283)
            loss = criterion(output, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                output = model(batch_x).squeeze(-1)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}'
        )

        # 保存验证集上损失最低的模型
        best_val_loss = save_best_model(model, optimizer, val_loss,
                                        best_val_loss)

    print("训练完成.")

    # 加载验证集损失最低的模型并进行推理
    load_best_model(model, optimizer)

    results = []
    model.eval()

    with torch.no_grad():
        for i in range(0, data_train_reshaped.size(0), batch_size):
            batch = data_train_reshaped[i:i + batch_size].cuda()
            output = model(batch)
            results.append(output.cpu())

    predicted_targets = torch.cat(results, dim=0).numpy()
    np.save("predicted_targets.npy", predicted_targets)

    print(predicted_targets.shape)
    print("预测结果已保存.")


if __name__ == "__main__":
    main()
