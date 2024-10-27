import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn
import os
import pandas as pd


def postprocessing(pred_array, wavelengths, index, sigma_pred):
    """Create a submission dataframe from its components
    
    Parameters:
    pred_array: ndarray of shape (n_samples, 283)
    index: pandas.Index of length n_samples with name 'planet_id'
    sigma_pred: float
    
    Return value:
    df: DataFrame of shape (n_samples, 566) with planet_id as index
    """
    return pd.concat([
        pd.DataFrame(pred_array.clip(0, None),
                     index=index,
                     columns=wavelengths.columns),
        pd.DataFrame(sigma_pred,
                     index=index,
                     columns=[f"sigma_{i}" for i in range(1, 284)])
    ],
                     axis=1)


# 保存与加载模型函数
def save_best_model(
    model,
    optimizer,
    val_loss,
    best_val_loss,
    path='best_model.pth',
    target_min=0,
    target_max=1,
):
    if val_loss < best_val_loss:
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': val_loss,
                'target_min': target_min,
                'target_max': target_max,
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
    return model, optimizer, best_val_loss, target_min, target_max


def load_traget(target_path):
    train_solution = np.loadtxt(target_path, delimiter=',', skiprows=1)
    targets = train_solution[:, 1:]
    targets_tensor = torch.tensor(targets).float()
    target_min = targets_tensor.min()
    target_max = targets_tensor.max()
    targets_normalized = (targets_tensor - target_min) / (target_max -
                                                          target_min)
    return targets_normalized, target_min, target_max


def load_AIRS_FGS_data(AIRs_path, FGS_path):
    signal_AIRS = np.load(AIRs_path)
    signal_FGS = np.load(FGS_path)
    FGS_column = signal_FGS.sum(axis=2)
    del signal_FGS
    dataset = np.concatenate([FGS_column[:, :, np.newaxis, :], signal_AIRS],
                             axis=2)
    del signal_AIRS, FGS_column
    return dataset


def normalized_reshaped(dataset):
    data_train_tensor = torch.tensor(dataset).float()
    del dataset
    data_min = data_train_tensor.min(dim=1, keepdim=True)[0]
    data_max = data_train_tensor.max(dim=1, keepdim=True)[0]
    data_train_normalized = (data_train_tensor - data_min) / (data_max -
                                                              data_min)
    del data_train_tensor
    data_train_normalized = data_train_normalized.permute(0, 2, 1, 3)

    return data_train_normalized


def train_fuc(data_train_reshaped,
              targets_normalized,
              model,
              optimizer,
              modelname,
              best_val_loss,
              train_epchos=50,
              batch_size=16,
              target_min=0,
              target_max=1):

    # 数据集拆分
    num_planets = data_train_reshaped.size(0)
    train_size = int(0.8 * num_planets)
    val_size = num_planets - train_size

    # 设置随机数种子
    seed = 42
    torch.manual_seed(seed)
    train_data, val_data = random_split(
        list(zip(data_train_reshaped, targets_normalized)),
        [train_size, val_size])

    # 数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    # 训练循环
    for epoch in range(train_epchos):
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
                                        path='best_model_' + modelname +
                                        '.pth',
                                        target_min=target_min,
                                        target_max=target_max)
    print("训练完成")


def predict_fuc(model, predict_data, batch_size=1):

    with torch.no_grad():
        model.eval()
        results = []

        for i in range(0, predict_data.size(0), batch_size):
            batch = predict_data[i:i + batch_size].cuda()
            output = model(batch).cpu().numpy()
            results.append(output)

            torch.cuda.empty_cache()  # 每次批次处理后清理显存

        # 合并所有批次结果
        all_predictions = torch.tensor(np.concatenate(results, axis=0))

    return all_predictions


def train_predict(ModelClass, modelname, batch_size, train_epchos):
    "数据加载与预处理"
    # 特征
    data_folder = 'input/binned-dataset-v3/'
    dataset = load_AIRS_FGS_data(f'{data_folder}/data_train.npy',
                                 f'{data_folder}/data_train_FGS.npy')
    data_train_reshaped = normalized_reshaped(dataset)

    # 目标
    auxiliary_folder = 'input/ariel-data-challenge-2024/'
    targets_normalized, target_min, target_max = load_traget(
        f'{auxiliary_folder}/train_labels.csv')

    # 初始化模型、损失函数和优化器
    model = ModelClass().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    """新模型修改模型名称"""
    try:
        model, optimizer, best_val_loss, target_min, target_max = load_best_model(
            model, optimizer, modelname)
    except:
        print("未找到最佳模型，开始训练。")
    "训练"
    train_fuc(data_train_reshaped,
              targets_normalized,
              model,
              optimizer,
              modelname,
              best_val_loss,
              train_epchos=train_epchos,
              batch_size=batch_size,
              target_min=target_min,
              target_max=target_max)
    "预测"
    model, optimizer, best_val_loss, target_min, target_max = load_best_model(
        model, optimizer, path='best_model_' + modelname + '.pth')

    all_predictions = predict_fuc(model, data_train_reshaped, batch_size)
    all_predictions = all_predictions * (target_max - target_min) + target_min
    all_predictions = all_predictions.numpy()
    np.save('predicted_targets_' + modelname + '.npy', all_predictions)
