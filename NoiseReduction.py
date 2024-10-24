import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

data_folder = 'input/binned-dataset-v3/'  # path to the folder containing the data
signal_AIRS_diff_transposed_binned = np.load(f'{data_folder}/data_train.npy')
signal_FGS_diff_transposed_binned = np.load(
    f'{data_folder}/data_train_FGS.npy')

FGS_column = signal_FGS_diff_transposed_binned.sum(axis=2)
del signal_FGS_diff_transposed_binned
dataset = np.concatenate(
    [signal_AIRS_diff_transposed_binned, FGS_column[:, :, np.newaxis, :]],
    axis=2)
del signal_AIRS_diff_transposed_binned, FGS_column

data_train_tensor = torch.tensor(dataset)
# 计算数据的最小值和最大值（用于归一化）
data_min = data_train_tensor.min(dim=1, keepdim=True)[0]  # (673, 1, 283, 32)
data_max = data_train_tensor.max(dim=1, keepdim=True)[0]  # (673, 1, 283, 32)

# 归一化到 [0, 1] 范围
data_train_normalized = (data_train_tensor - data_min) / (data_max - data_min)
del data_train_tensor, dataset
# 假设你的数据是 data_train，形状为 (673, 187, 283, 32)
# 重塑数据：每个波长的时间序列作为输入 (673 * 283, 187, 32)
data_train_reshaped = data_train_normalized.permute(0, 2, 1,
                                                    3).reshape(-1, 187,
                                                               32).float()
del data_train_normalized

# 数据集与数据加载器
batch_size = 256
dataset2 = TensorDataset(data_train_reshaped, data_train_reshaped)  # 输入和目标相同
data_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=True)


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


class DenoisingTransformer(nn.Module):

    def __init__(self, d_model=32, nhead=8, num_layers=4, dropout=0.1):
        super(DenoisingTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)
        self.decoder = nn.Linear(d_model, d_model)  # 输出层

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


# 初始化模型
model = DenoisingTransformer().cuda()

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

try:
    load_checkpoint(model, optimizer)
except FileNotFoundError:
    print("No checkpoint found, starting from scratch.")

# 训练循环
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_x, batch_y in data_loader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        # 前向传播
        optimizer.zero_grad()
        output = model(batch_x)

        # 计算损失并反向传播
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.5f}'
    )
    save_checkpoint(model, optimizer)

save_checkpoint(model, optimizer)

# 推理时按批次处理，减少 GPU 显存占用
batch_size = 64  # 根据 GPU 显存调整合适的 batch size
results = []

model.eval()
with torch.no_grad():
    for i in range(0, data_train_reshaped.shape[0], batch_size):
        # 提取一个批次的数据并移动到 GPU
        batch = data_train_reshaped[i:i + batch_size].cuda()

        # 进行推理
        output = model(batch)

        # 将结果移回 CPU 并存储
        results.append(output.cpu())
        torch.cuda.empty_cache()

# 合并所有批次结果
denoised_data = torch.cat(results, dim=0)
del results

# 恢复原始数据的形状
denoised_data = denoised_data.reshape(673, 283, 187, 32).permute(0, 2, 1, 3)
data_restored = denoised_data * (data_max - data_min) + data_min
del denoised_data
# 打印结果示例
print(data_restored.shape)  # 应为 (673, 187, 283, 32)
data_restored_np = data_restored.cpu().numpy()
np.save("denoised_data.npy", data_restored_np)
