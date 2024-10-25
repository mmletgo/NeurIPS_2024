import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


# 模型保存与加载函数
def save_checkpoint(model,
                    optimizer,
                    path='noise_reduction_model_checkpoint.pth'):
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
    print("Checkpoint saved.")


def load_checkpoint(model,
                    optimizer,
                    path='noise_reduction_model_checkpoint.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Checkpoint loaded.")


def main():
    # 读取目标文件
    auxiliary_folder = 'input/ariel-data-challenge-2024/'
    train_solution = np.loadtxt(f'{auxiliary_folder}/train_labels.csv',
                                delimiter=',',
                                skiprows=1)
    targets = train_solution[:, 1:]  # 形状 (673, 283)
    print(f"目标数据读取完毕，形状: {targets.shape}")

    # 数据加载
    data_folder = 'input/binned-dataset-v3/'
    signal_AIRS_diff_transposed_binned = np.load(
        f'{data_folder}/data_train.npy')
    signal_FGS_diff_transposed_binned = np.load(
        f'{data_folder}/data_train_FGS.npy')
    print("光谱数据初步加载完毕.")

    # 数据处理与组合
    FGS_column = signal_FGS_diff_transposed_binned.sum(axis=2)
    del signal_FGS_diff_transposed_binned

    dataset = np.concatenate(
        [FGS_column[:, :, np.newaxis, :], signal_AIRS_diff_transposed_binned],
        axis=2)
    del signal_AIRS_diff_transposed_binned, FGS_column

    # 转换为 Tensor 并归一化
    data_train_tensor = torch.tensor(dataset).float()
    data_min = data_train_tensor.min(dim=1, keepdim=True)[0]
    data_max = data_train_tensor.max(dim=1, keepdim=True)[0]
    data_train_normalized = (data_train_tensor - data_min) / (data_max -
                                                              data_min)
    del data_train_tensor, dataset  # 释放内存

    # 确保数据为 4D 形状 (673, 283, 187, 32)
    data_train_reshaped = data_train_normalized.permute(0, 2, 1, 3)
    del data_train_normalized  # 释放内存
    print(f"数据形状: {data_train_reshaped.shape}")

    # 创建数据集和加载器
    batch_size = 32
    targets_tensor = torch.tensor(targets).float()
    dataset2 = TensorDataset(data_train_reshaped, targets_tensor)
    data_loader = DataLoader(dataset2,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)
    print("数据加载器准备完毕.")

    # 初始化模型、损失函数和优化器
    model = EnhancedTimeWavelengthTransformer().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    try:
        load_checkpoint(model, optimizer)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())

    # 训练循环
    num_epochs = 50
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            optimizer.zero_grad()

            # 前向传播和损失计算
            output = model(batch_x).squeeze(-1)  # 输出形状 (batch_size, 283)
            loss = criterion(output, batch_y)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            torch.cuda.empty_cache()  # 每次批次处理后清理显存

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.5f}'
        )
        save_checkpoint(model, optimizer)

    # 保存最终模型
    save_checkpoint(model, optimizer)
    print("训练完成.")

    # 推理阶段
    results = []
    model.eval()

    with torch.no_grad():
        for i in range(0, data_train_reshaped.shape[0], batch_size):
            batch = data_train_reshaped[i:i + batch_size].cuda()
            output = model(batch)
            results.append(output.cpu())
            torch.cuda.empty_cache()

    # 合并推理结果
    predicted_targets = torch.cat(results, dim=0).numpy()

    # 保存预测结果
    np.save("predicted_targets.npy", predicted_targets)
    print(predicted_targets.shape)  # 应为 (673, 283)
    print("预测结果已保存.")


# 程序入口
if __name__ == "__main__":
    main()
