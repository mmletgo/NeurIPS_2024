import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torch import nn
import pandas as pd
import scipy.stats
from scipy.optimize import minimize
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold


class ParticipantVisibleError(Exception):
    pass


def competition_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    naive_mean: float,
    naive_sigma: float,
    sigma_true: float,
    row_id_column_name='planet_id',
) -> float:
    '''
    This is a Gaussian Log Likelihood based metric. For a submission, which contains the predicted mean (x_hat) and variance (x_hat_std),
    we calculate the Gaussian Log-likelihood (GLL) value to the provided ground truth (x). We treat each pair of x_hat,
    x_hat_std as a 1D gaussian, meaning there will be 283 1D gaussian distributions, hence 283 values for each test spectrum,
    the GLL value for one spectrum is the sum of all of them.

    Inputs:
        - solution: Ground Truth spectra (from test set)
            - shape: (nsamples, n_wavelengths)
        - submission: Predicted spectra and errors (from participants)
            - shape: (nsamples, n_wavelengths*2)
        naive_mean: (float) mean from the train set.
        naive_sigma: (float) standard deviation from the train set.
        sigma_true: (float) essentially sets the scale of the outputs.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if submission.min().min() < 0:
        raise ParticipantVisibleError('Negative values in the submission')
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(
                f'Submission column {col} must be a number')

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != n_wavelengths * 2:
        raise ParticipantVisibleError(
            'Wrong number of columns in the submission')

    y_pred = submission.iloc[:, :n_wavelengths].values
    # Set a non-zero minimum sigma pred to prevent division by zero errors.
    sigma_pred = np.clip(submission.iloc[:, n_wavelengths:].values,
                         a_min=10**-15,
                         a_max=None)
    y_true = solution.values

    GLL_pred = np.sum(
        scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred))
    GLL_true = np.sum(
        scipy.stats.norm.logpdf(y_true,
                                loc=y_true,
                                scale=sigma_true * np.ones_like(y_true)))
    GLL_mean = np.sum(
        scipy.stats.norm.logpdf(y_true,
                                loc=naive_mean * np.ones_like(y_true),
                                scale=naive_sigma * np.ones_like(y_true)))

    submit_score = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)
    return float(np.clip(submit_score, 0.0, 1.0))


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


def train_func(data_train_reshaped,
               targets_normalized,
               model,
               optimizer,
               modelname,
               best_val_loss,
               train_epochs=50,
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
    for epoch in range(train_epochs):
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
                all_predictions = valid_output.cpu() * (
                    target_max - target_min) + target_min
                all_predictions = all_predictions.numpy()
                targets = valid_batch_y.cpu() * (target_max -
                                                 target_min) + target_min
                targets = targets.numpy()
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


def train_func_2input(data_train_reshaped,
                      peak_areas,
                      targets_normalized,
                      model,
                      optimizer,
                      modelname,
                      best_val_loss,
                      train_epochs=50,
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

    # 将数据打包为 (光通量数据, 吸收峰面积, 目标值) 的元组
    dataset = list(zip(data_train_reshaped, peak_areas, targets_normalized))
    train_data, val_data = random_split(dataset, [train_size, val_size])

    # 数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(train_epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_peak, batch_y in train_loader:
            batch_x, batch_peak, batch_y = (batch_x.cuda(), batch_peak.cuda(),
                                            batch_y.cuda())
            optimizer.zero_grad()

            # 将两个输入传递给模型
            output = model(batch_x, batch_peak).squeeze(-1)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            torch.cuda.empty_cache()

        train_loss /= len(train_loader)

        # 验证模型
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for valid_batch_x, valid_batch_peak, valid_batch_y in val_loader:
                valid_batch_x, valid_batch_peak, valid_batch_y = (
                    valid_batch_x.cuda(), valid_batch_peak.cuda(),
                    valid_batch_y.cuda())

                # 验证时也传递两个输入
                valid_output = model(valid_batch_x,
                                     valid_batch_peak).squeeze(-1)
                val_loss += criterion(valid_output, valid_batch_y).item()

                all_predictions = valid_output.cpu() * (
                    target_max - target_min) + target_min
                all_predictions = all_predictions.numpy()
                targets = valid_batch_y.cpu() * (target_max -
                                                 target_min) + target_min
                targets = targets.numpy()

                torch.cuda.empty_cache()

        val_loss /= len(val_loader)

        print(
            f'Epoch [{epoch + 1}], Train Loss: {train_loss:.16f}, Val Loss: {val_loss:.16f}'
        )

        best_val_loss = save_best_model(model,
                                        optimizer,
                                        val_loss,
                                        best_val_loss,
                                        path=f'best_model_{modelname}.pth',
                                        target_min=target_min,
                                        target_max=target_max)

    print("训练完成")


def predict_func_2input(model, predict_data_x, peak_areas, batch_size=1):
    with torch.no_grad():
        model.eval()
        results = []

        # 遍历数据集，按批次进行预测
        for i in range(0, predict_data_x.size(0), batch_size):
            batch_x = predict_data_x[i:i + batch_size].cuda()
            batch_peak = peak_areas[i:i + batch_size].cuda()

            # 将两个输入传递给模型
            output = model(batch_x, batch_peak).cpu().numpy()
            results.append(output)

            torch.cuda.empty_cache()  # 每次批次处理后清理显存

        # 合并所有批次的预测结果
        all_predictions = torch.tensor(np.concatenate(results, axis=0))

    return all_predictions


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


def train_predict(ModelClass, modelname, batch_size, train_epochs):
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
            model, optimizer, path='best_model_' + modelname + '.pth')
    except:
        print("未找到最佳模型，开始训练。")
    "训练"
    train_func(data_train_reshaped,
               targets_normalized,
               model,
               optimizer,
               modelname,
               best_val_loss,
               train_epochs=train_epochs,
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


def phase_detector(signal):
    MIN = np.argmin(signal[30:140]) + 30  # 在索引区间 [30, 140) 中找到最小值的位置，加上偏移量 30
    signal1 = signal[:MIN]  # 分割信号，获取前半部分
    signal2 = signal[MIN:]  # 获取后半部分

    # 计算前半部分的梯度（即一阶导数），并归一化
    first_derivative1 = np.gradient(signal1)
    first_derivative1 /= first_derivative1.max()
    first_derivative2 = np.gradient(signal2)  # 计算后半部分的梯度，并归一化
    first_derivative2 /= first_derivative2.max()

    # 找到前半部分梯度的最小值位置作为 phase1
    phase1 = np.argmin(first_derivative1)

    # 找到后半部分梯度的最大值位置作为 phase2，并加上 MIN 偏移量
    phase2 = np.argmax(first_derivative2) + MIN

    # 返回两个阶段的索引
    return phase1, phase2


def predict_spectra(signal):

    def objective_to_minimize(s):
        delta = 2  # 用于平滑的偏移量
        power = 3  # 多项式拟合的阶
        x = list(range(signal.shape[0] - delta * 4))  # 拟合的 x 坐标范围

        # 构建 y 坐标，信号分为三段，并对两边部分应用s的缩放
        y = ((signal[:phase1 - delta] * (1 - s)).tolist() +
             signal[phase1 + delta:phase2 - delta].tolist() +
             (signal[phase2 + delta:] * (1 - s)).tolist())

        # 对构建的信号 y 进行三阶多项式拟合
        z = np.polyfit(x, y, deg=power)
        p = np.poly1d(z)  # 将拟合结果转为多项式对象

        # 计算拟合曲线与原信号之间的平均绝对误差
        q = np.abs(p(x) - y).mean()
        return q

    # 对信号的每一列取平均值
    signal = signal[:, 1:].mean(axis=1)

    # 使用 phase_detector 函数检测两个阶段的索引
    phase1, phase2 = phase_detector(signal)

    # 使用 Nelder-Mead 优化方法最小化误差，找到最佳的缩放参数 s
    s = minimize(fun=objective_to_minimize, x0=[0.0001],
                 method="Nelder-Mead").x[0]
    return s  # 返回优化后的缩放参数 s


def predict_y(x, coefficients):
    return np.polyval(coefficients, x)


def cal_flux(signal):
    delta = 2
    signal_w = signal[:, 1:].mean(axis=1)
    phase1, phase2 = phase_detector(signal_w)
    alphas = []
    for wave in range(len(signal[0])):
        flux_dropdown_sum = 0
        x = np.array(
            list(range(phase1 - delta)) +
            list(range(phase2 + delta, signal.shape[0])))
        y = np.array(signal[:phase1 - delta, wave].tolist() +
                     signal[phase2 + delta:, wave].tolist())
        coefficients = np.polyfit(x, y, 3)
        x_preditc = np.array(range(phase1, phase2 + 1))
        high = predict_y(x_preditc, coefficients)
        for i in range(phase2 + 1 - phase1):
            flux_dropdown_sum += (high[i] - signal[i + phase1, wave])
        alphas.append(flux_dropdown_sum)
    alphas = np.array(alphas)
    min_alpha = alphas.min()
    max_alpha = alphas.max()
    alphas = (alphas - min_alpha) / (max_alpha - min_alpha)
    return alphas


def train_predict2(ModelClass, modelname, batch_size, train_epochs):
    "数据加载与预处理"
    # 特征
    with open('input/train_preprocessed.pkl', 'rb') as file:
        full_predictions_spectra = pickle.load(file)
    full_whitelight_s_train = np.array([
        predict_spectra(full_predictions_spectra[i])
        for i in range(len(full_predictions_spectra))
    ])  # 预测每个星球的白光缩放比例S
    # 目标
    train_solution = np.loadtxt(
        'input/ariel-data-challenge-2024/train_labels.csv',
        delimiter=',',
        skiprows=1)
    targets = train_solution[:, 1:]
    newtarget = targets / full_whitelight_s_train[:, np.newaxis]
    targets_tensor = torch.tensor(newtarget).float()
    target_min = targets_tensor.min()
    target_max = targets_tensor.max()
    full_targets_normalized = (targets_tensor - target_min) / (target_max -
                                                               target_min)

    # 使用 K-means 聚类
    n_clusters = 20  # 你可以根据需要调整聚类数量
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=42).fit(full_targets_normalized)
    cluster_labels = kmeans.labels_

    np.random.seed(21)
    # 从每个聚类中随机抽取样本
    sampled_indices = []
    samples_per_cluster = 320 // n_clusters  # 每个聚类中抽取的样本数量
    for cluster in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) >= samples_per_cluster:
            sampled_indices.extend(
                np.random.choice(cluster_indices,
                                 samples_per_cluster,
                                 replace=False))
        else:
            sampled_indices.extend(cluster_indices)  # 如果样本数量不足，全部选取

    # 如果总样本数量不足 320，补充剩余的样本
    if len(sampled_indices) < 320:
        remaining_indices = np.setdiff1d(
            np.arange(full_predictions_spectra.shape[0]), sampled_indices)
        additional_samples = np.random.choice(remaining_indices,
                                              320 - len(sampled_indices),
                                              replace=False)
        sampled_indices.extend(additional_samples)

    predictions_spectra = full_predictions_spectra[sampled_indices]
    targets_normalized = full_targets_normalized[sampled_indices]

    data_train_reshaped = torch.tensor(predictions_spectra).float()

    # 初始化模型、损失函数和优化器
    model = ModelClass().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    """新模型修改模型名称"""
    try:
        model, optimizer, best_val_loss, target_min, target_max = load_best_model(
            model, optimizer, path='best_model_' + modelname + '.pth')
    except:
        print("未找到最佳模型，开始训练。")
    "训练"
    train_func(data_train_reshaped,
               targets_normalized,
               model,
               optimizer,
               modelname,
               best_val_loss,
               train_epochs=train_epochs,
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


def train_predict3(ModelClass, modelname, batch_size, train_epochs):
    "数据加载与预处理"
    # 特征
    with open('input/train_preprocessed.pkl', 'rb') as file:
        full_predictions_spectra = pickle.load(file)
    full_whitelight_s_train = np.array([
        predict_spectra(full_predictions_spectra[i])
        for i in range(len(full_predictions_spectra))
    ])  # 预测每个星球的白光缩放比例S
    full_light_alpha_train = np.array([
        cal_flux(full_predictions_spectra[i])
        for i in range(len(full_predictions_spectra))
    ])  # 计算每个星球的各个波段的吸收峰相对面积
    # 目标
    train_solution = np.loadtxt(
        'input/ariel-data-challenge-2024/train_labels.csv',
        delimiter=',',
        skiprows=1)
    targets = train_solution[:, 1:]
    newtarget = targets / full_whitelight_s_train[:, np.newaxis]
    targets_tensor = torch.tensor(newtarget).float()
    target_min = targets_tensor.min()
    target_max = targets_tensor.max()
    full_targets_normalized = (targets_tensor - target_min) / (target_max -
                                                               target_min)

    # 使用 K-means 聚类
    n_clusters = 16  # 根据需要调整聚类数量
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=42).fit(full_targets_normalized)
    cluster_labels = kmeans.labels_

    np.random.seed(21)
    # 初始化存储采样的索引列表
    sampled_indices = []
    samples_per_cluster = 640 // n_clusters  # 每个聚类中目标样本数量

    for cluster in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster)[0]

        # 如果样本数量不足，则进行上采样
        if len(cluster_indices) < samples_per_cluster:
            temp_samples = cluster_indices.tolist()
            sampled_indices.extend(temp_samples)
            remaining_samples = samples_per_cluster - len(cluster_indices)
            sampled_indices.extend(
                np.random.choice(cluster_indices,
                                 remaining_samples,
                                 replace=True))
        else:
            # 样本数量充足时正常采样
            sampled_indices.extend(
                np.random.choice(cluster_indices,
                                 samples_per_cluster,
                                 replace=False))

    predictions_spectra = full_predictions_spectra[sampled_indices]
    min_values = predictions_spectra.min(axis=(1, 2), keepdims=True)
    max_values = predictions_spectra.max(axis=(1, 2), keepdims=True)
    normalized_spectra = (predictions_spectra - min_values) / (max_values -
                                                               min_values)
    light_alpha_train = full_light_alpha_train[sampled_indices]
    targets_normalized = full_targets_normalized[sampled_indices]

    whitelight_s_train = full_whitelight_s_train[sampled_indices]
    whitelight_s_train_expanded = np.expand_dims(whitelight_s_train * 100,
                                                 axis=1)
    combined_array = np.concatenate(
        (light_alpha_train, whitelight_s_train_expanded), axis=1)
    combined_array.shape

    light_alpha_train = torch.tensor(combined_array).float()
    data_train_reshaped = torch.tensor(normalized_spectra).float()

    # 初始化模型、损失函数和优化器
    model = ModelClass().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    """新模型修改模型名称"""
    try:
        model, optimizer, best_val_loss, target_min, target_max = load_best_model(
            model, optimizer, path='best_model_' + modelname + '.pth')
    except:
        print("未找到最佳模型，开始训练。")
    "训练"
    train_func_2input(data_train_reshaped,
                      light_alpha_train,
                      targets_normalized,
                      model,
                      optimizer,
                      modelname,
                      best_val_loss,
                      train_epochs=train_epochs,
                      batch_size=batch_size,
                      target_min=target_min,
                      target_max=target_max)
    "预测"
    model, optimizer, best_val_loss, target_min, target_max = load_best_model(
        model, optimizer, path='best_model_' + modelname + '.pth')

    all_predictions = predict_func_2input(model, data_train_reshaped,
                                          light_alpha_train, batch_size)
    all_predictions = all_predictions * (target_max - target_min) + target_min
    all_predictions = all_predictions.numpy()
    np.save('predicted_targets_' + modelname + '.npy', all_predictions)
