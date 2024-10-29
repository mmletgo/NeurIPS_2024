import pandas as pd
import numpy as np
from pqdm.processes import pqdm
import itertools
from astropy.stats import sigma_clip
import pickle


# 传感器信号的校准。它处理暗电流、死像素、平场校准和线性校准。
class Calibrator:
    cut_inf = 39  # 下限：切割信号的起始位置
    cut_sup = 321  # 上限：切割信号的结束位置

    # 定义每种传感器的信号尺寸 (原始尺寸和切割后的尺寸)
    sensor_to_sizes_dict = {
        "AIRS-CH0": [[11250, 32, 356], [1, 32, cut_sup - cut_inf]],
        "FGS1": [[135000, 32, 32], [1, 32, 32]],
    }

    # 每个传感器的线性校正矩阵尺寸
    sensor_to_linear_corr_dict = {
        "AIRS-CH0": (6, 32, 356),
        "FGS1": (6, 32, 32)
    }

    def __init__(self, dataset, planet_id, sensor):
        # 初始化类，传入数据集、行星 ID 和传感器类型
        self.dataset = dataset
        self.planet_id = planet_id
        self.sensor = sensor

    def _apply_linear_corr(self, linear_corr, clean_signal):
        # 翻转线性校正矩阵的第一个轴
        linear_corr = np.flip(linear_corr, axis=0)

        # 遍历信号的每个像素，并应用多项式校正
        for x, y in itertools.product(range(clean_signal.shape[1]),
                                      range(clean_signal.shape[2])):
            poli = np.poly1d(linear_corr[:, x, y])  # 创建多项式对象
            clean_signal[:, x, y] = poli(clean_signal[:, x, y])  # 校正信号值
        return clean_signal

    def _clean_dark(self, signal, dark, dt):
        # 重复暗帧，以匹配信号的时间轴
        dark = np.tile(dark, (signal.shape[0], 1, 1))

        # 减去暗电流的影响，dt 是时间差
        signal -= dark * dt[:, np.newaxis, np.newaxis]
        return signal

    def get_calibrated_signal(self):

        # 从 parquet 文件读取信号数据和校准数据
        signal = pd.read_parquet(
            f"/kaggle/input/ariel-data-challenge-2024/{self.dataset}/{self.planet_id}/{self.sensor}_signal.parquet"
        ).to_numpy()
        dark_frame = pd.read_parquet(
            f"/kaggle/input/ariel-data-challenge-2024/{self.dataset}/{self.planet_id}/{self.sensor}_calibration/dark.parquet",
            engine="pyarrow",
        ).to_numpy()
        dead_frame = pd.read_parquet(
            f"/kaggle/input/ariel-data-challenge-2024/{self.dataset}/{self.planet_id}/{self.sensor}_calibration/dead.parquet",
            engine="pyarrow",
        ).to_numpy()
        flat_frame = pd.read_parquet(
            f"/kaggle/input/ariel-data-challenge-2024/{self.dataset}/{self.planet_id}/{self.sensor}_calibration/flat.parquet",
            engine="pyarrow",
        ).to_numpy()

        # 读取并调整线性校正矩阵
        linear_corr = (pd.read_parquet(
            f"/kaggle/input/ariel-data-challenge-2024/{self.dataset}/{self.planet_id}/{self.sensor}_calibration/linear_corr.parquet"
        ).values.astype(np.float64).reshape(
            self.sensor_to_linear_corr_dict[self.sensor]))

        # 调整信号的尺寸
        signal = signal.reshape(self.sensor_to_sizes_dict[self.sensor][0])
        adc_info = pd.read_csv(
            "/kaggle/input/ariel-data-challenge-2024/" +
            f"{self.dataset}_adc_info.csv",
            index_col="planet_id",
        )
        # 根据增益和偏移进行归一化
        gain = adc_info.loc[self.planet_id, f"{self.sensor}_adc_gain"]
        offset = adc_info.loc[self.planet_id, f"{self.sensor}_adc_offset"]
        signal = signal / gain + offset

        # 使用 sigma_clip 检测并掩盖热点像素
        hot = sigma_clip(dark_frame, sigma=5, maxiters=5).mask
        print(f"Hot pixel mask shape: {hot.shape}")

        # 针对 AIRS-CH0 传感器进行数据裁剪
        if self.sensor == "AIRS-CH0":
            signal = signal[:, :, self.cut_inf:self.cut_sup]
            dt = np.ones(len(signal)) * 0.1
            dt[1::2] += 4.5  # @bilzard idea

            # 对校准帧和掩码进行裁剪
            linear_corr = linear_corr[:, :, self.cut_inf:self.cut_sup]
            dark_frame = dark_frame[:, self.cut_inf:self.cut_sup]
            dead_frame = dead_frame[:, self.cut_inf:self.cut_sup]
            flat_frame = flat_frame[:, self.cut_inf:self.cut_sup]
            hot = hot[:, self.cut_inf:self.cut_sup]

        # 针对 FGS1 传感器的时间差调整
        elif self.sensor == "FGS1":
            dt = np.ones(len(signal)) * 0.1
            dt[1::2] += 0.1
        # 将信号中的负值裁剪为 0
        signal = signal.clip(0)  # @graySnow idea

        # 应用线性校正和暗电流去除
        linear_corr_signal = self._apply_linear_corr(linear_corr, signal)
        signal = self._clean_dark(linear_corr_signal, dark_frame, dt)

        # 处理平场校准，并应用 NaN 掩码
        flat = flat_frame.reshape(self.sensor_to_sizes_dict[self.sensor][1])
        flat[dead_frame.reshape(
            self.sensor_to_sizes_dict[self.sensor][1])] = np.nan
        flat[hot.reshape(self.sensor_to_sizes_dict[self.sensor][1])] = np.nan

        # 应用平场校准
        signal = signal / flat
        return signal  # 返回校准后的信号


class Preprocessor:

    # 传感器到分组大小的映射
    sensor_to_binning = {"AIRS-CH0": 30, "FGS1": 30 * 12}

    # 定义每个传感器的分组后的数据尺寸
    sensor_to_binned_dict = {
        "AIRS-CH0": [11250 // sensor_to_binning["AIRS-CH0"] // 2, 282],
        "FGS1": [135000 // sensor_to_binning["FGS1"] // 2],
    }

    def __init__(self, dataset, planet_id, sensor):

        # 初始化类，传入数据集、行星 ID 和传感器类型
        self.dataset = dataset
        self.planet_id = planet_id
        self.sensor = sensor
        self.binning = self.sensor_to_binning[sensor]  # 获取传感器的分组大小

    def preprocess_signal(self):
        # 获取校准后的信号
        signal = Calibrator(dataset=self.dataset,
                            planet_id=self.planet_id,
                            sensor=self.sensor).get_calibrated_signal()

        # 针对不同传感器的信号进行裁剪
        if self.sensor == "AIRS-CH0":
            signal = signal[:, 10:22, :]
        elif self.sensor == "FGS1":
            signal = signal[:, 10:22, 10:22]
            signal = signal.reshape(signal.shape[0],
                                    signal.shape[1] * signal.shape[2])
        # 计算每个时间步的均值信号
        mean_signal = np.nanmean(signal, axis=1)
        cds_signal = mean_signal[1::2] - mean_signal[0::2]  # 计算差分信号

        # 对信号进行分组并求平均
        binned = np.zeros((self.sensor_to_binned_dict[self.sensor]))
        for j in range(cds_signal.shape[0] // self.binning):
            binned[j] = cds_signal[j * self.binning:j * self.binning +
                                   self.binning].mean(axis=0)

        # 如果是 FGS1 传感器，将结果 reshape
        if self.sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        return binned  # 返回分组后的信号


def preprocessor(x):
    return Preprocessor(**x).preprocess_signal()  # 使用传入的参数创建预处理器并处理信号


def preprocess_pro(dataset="train"):
    adc_info = pd.read_csv(
        "/kaggle/input/ariel-data-challenge-2024/" + f"{dataset}_adc_info.csv",
        index_col="planet_id",
    )

    planet_ids = adc_info.index

    args_fgs1 = [
        dict(dataset=dataset, planet_id=planet_id, sensor="FGS1")  # FGS1 传感器信号
        for planet_id in planet_ids
    ]
    preprocessed_signal_fgs1 = pqdm(args_fgs1, preprocessor, n_jobs=4)  # 并行预处理

    args_airs_ch0 = [
        dict(dataset=dataset, planet_id=planet_id,
             sensor="AIRS-CH0")  # AIRS-CH0 传感器信号
        for planet_id in planet_ids
    ]
    preprocessed_signal_airs_ch0 = pqdm(args_airs_ch0, preprocessor,
                                        n_jobs=4)  # 并行预处理

    preprocessed_signal = np.concatenate(
        [
            np.stack(preprocessed_signal_fgs1),
            np.stack(preprocessed_signal_airs_ch0)
        ],
        axis=2  # 将两个传感器的信号按第 3 维度（axis=2）进行合并，以便后续一起处理
    )
    pickle.dump(preprocessed_signal, open(f"{dataset}_preprocessed.pkl",
                                          "wb"))  # 保存预处理后的信号
    return preprocessed_signal


if __name__ == "__main__":
    preprocess_pro("train")  # 预处理训练集
