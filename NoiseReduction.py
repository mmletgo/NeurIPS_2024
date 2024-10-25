import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
        self.decoder = nn.Linear(d_model, d_model)

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
        return self.decoder(x)


# 主函数
def main():
    # 数据加载
    data_folder = 'input/binned-dataset-v3/'
    signal_AIRS_diff_transposed_binned = np.load(
        f'{data_folder}/data_train.npy')
    signal_FGS_diff_transposed_binned = np.load(
        f'{data_folder}/data_train_FGS.npy')
    print("数据初步加载完毕.")

    # 数据处理与组合
    FGS_column = signal_FGS_diff_transposed_binned.sum(axis=2)
    del signal_FGS_diff_transposed_binned

    dataset = np.concatenate(
        [signal_AIRS_diff_transposed_binned, FGS_column[:, :, np.newaxis, :]],
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

    # 数据加载器
    batch_size = 16  # 批次大小减小
    dataset2 = TensorDataset(data_train_reshaped, data_train_reshaped)
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
            output = model(batch_x)
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
            torch.cuda.empty_cache()  # 推理时每次批次处理后清理显存

    # 合并推理结果
    denoised_data = torch.cat(results, dim=0)
    del results

    # 恢复原始数据形状
    denoised_data = denoised_data.reshape(673, 283, 187,
                                          32).permute(0, 2, 1, 3)
    data_restored = denoised_data * (data_max - data_min) + data_min
    del denoised_data

    # 保存最终降噪数据
    data_restored_np = data_restored.cpu().numpy()
    np.save("denoised_data.npy", data_restored_np)

    print(data_restored.shape)  # 应为 (673, 187, 283, 32)
    print("降噪数据已保存.")


# 入口点
if __name__ == "__main__":
    main()
