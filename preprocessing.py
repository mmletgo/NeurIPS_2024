import numpy as np
import itertools
import os
import pandas as pd
from tqdm import tqdm
import torch


def normalize_and_reshape_chunks(dataset_test, planet_num):

    # 创建内存映射文件用于存储归一化和重塑后的数据
    normalized_reshaped_test = torch.empty((planet_num, 283, 187, 32),
                                           dtype=torch.float32)

    for i in range(planet_num):
        data_test_tensor = torch.tensor(dataset_test[i]).float()
        data_min = data_test_tensor.min(dim=(0, 1), keepdim=True)[0]
        data_max = data_test_tensor.max(dim=(0, 1), keepdim=True)[0]
        data_test_normalized = (data_test_tensor - data_min) / (data_max -
                                                                data_min)
        del data_test_tensor
        data_test_reshaped = data_test_normalized.permute(1, 0, 2)
        del data_test_normalized
        normalized_reshaped_test[i] = data_test_reshaped
        del data_test_reshaped

    return normalized_reshaped_test


def load_data_chunked(file, nb_files):
    for i in range(nb_files):
        file_path = file + '_{}.npy'.format(i)
        data = np.load(file_path)
        yield data
        os.remove(file_path)  # 删除文件


def del_files(file, nb_files):
    for i in range(nb_files):
        file_path = file + '_{}.npy'.format(i)
        if os.path.exists(file_path):
            os.remove(file_path)


def process_and_memmap_chunks(path_out, nb_files, chunk_size, output_file):
    # 创建内存映射文件
    dataset_test = np.memmap(output_file,
                             dtype='float32',
                             mode='w+',
                             shape=(nb_files * chunk_size, 187, 283, 32))

    current_index = 0

    for test_signal_FGS_chunk, test_signal_AIRS_chunk in zip(
            load_data_chunked(path_out + 'FGS1_test', nb_files),
            load_data_chunked(path_out + 'AIRS_clean_test', nb_files)):

        test_FGS_column_chunk = test_signal_FGS_chunk.sum(axis=2)
        del test_signal_FGS_chunk
        combined_chunk = np.concatenate([
            test_FGS_column_chunk[:, :, np.newaxis, :], test_signal_AIRS_chunk
        ],
                                        axis=2)
        del test_FGS_column_chunk, test_signal_AIRS_chunk

        # 将处理后的数据写入内存映射文件
        dataset_test[current_index:current_index +
                     combined_chunk.shape[0]] = combined_chunk
        current_index += combined_chunk.shape[0]
        del combined_chunk

    return dataset_test


def sigma_clip(data, sigma=3, max_iters=5):
    """
    Perform sigma clipping on the input data and return a mask of clipped values.
    
    Parameters:
    - data: numpy array of data to be sigma clipped
    - sigma: number of standard deviations to use for clipping
    - max_iters: maximum number of iterations
    
    Returns:
    - mask: numpy array with True for clipped values and False for retained values
    """
    data = np.asarray(data)
    clipped_data = data.copy()
    mask = np.zeros(data.shape, dtype=bool)

    for _ in range(max_iters):
        mean = np.nanmean(clipped_data)
        std = np.nanstd(clipped_data)
        new_mask = np.abs(clipped_data - mean) > sigma * std
        if not np.any(new_mask):
            break
        clipped_data[new_mask] = np.nan
        mask = mask | new_mask

    return mask


def ADC_convert(signal, gain, offset):
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal


def mask_hot_dead(signal, dead, dark):
    hot = sigma_clip(dark, sigma=5, max_iters=5)
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)
    return signal


def apply_linear_corr(linear_corr, clean_signal):
    linear_corr = np.flip(linear_corr, axis=0)
    for x, y in itertools.product(range(clean_signal.shape[1]),
                                  range(clean_signal.shape[2])):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    return clean_signal


def clean_dark(signal, dead, dark, dt):

    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))

    signal -= dark * dt[:, np.newaxis, np.newaxis]
    return signal


def get_cds(signal):
    cds = signal[:, 1::2, :, :] - signal[:, ::2, :, :]
    return cds


def bin_obs(cds_signal, binning):
    cds_transposed = cds_signal.transpose(0, 1, 3, 2)
    cds_binned = np.zeros(
        (cds_transposed.shape[0], cds_transposed.shape[1] // binning,
         cds_transposed.shape[2], cds_transposed.shape[3]))
    for i in range(cds_transposed.shape[1] // binning):
        cds_binned[:, i, :, :] = np.sum(cds_transposed[:, i * binning:(i + 1) *
                                                       binning, :, :],
                                        axis=1)
    return cds_binned


def correct_flat_field(flat, dead, signal):
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat
    return signal


def get_index(files, CHUNKS_SIZE):
    index = []
    for file in files:
        file_name = file.split('/')[-1]
        if file_name.split('_')[0] == 'AIRS-CH0' and file_name.split(
                '_')[1] == 'signal.parquet':
            file_index = os.path.basename(os.path.dirname(file))
            index.append(int(file_index))
    index = np.array(index)
    index = np.sort(index)
    # credit to DennisSakva
    index = np.array_split(index, len(index) // CHUNKS_SIZE)

    return index


def preprocess(path_folder, path_out, index, CHUNKS_SIZE=1):
    test_adc_info = pd.read_csv(os.path.join(path_folder, 'test_adc_info.csv'))
    test_adc_info = test_adc_info.set_index('planet_id')
    axis_info = pd.read_parquet(os.path.join(path_folder, 'axis_info.parquet'))
    DO_MASK = True
    DO_THE_NL_CORR = False
    DO_DARK = True
    DO_FLAT = True
    TIME_BINNING = True

    cut_inf, cut_sup = 39, 321
    l = cut_sup - cut_inf

    for n, index_chunk in enumerate(tqdm(index)):
        AIRS_CH0_clean = np.zeros((CHUNKS_SIZE, 11250, 32, l))
        FGS1_clean = np.zeros((CHUNKS_SIZE, 135000, 32, 32))

        for i in range(CHUNKS_SIZE):
            # Load and process AIRS-CH0 signal
            df = pd.read_parquet(
                os.path.join(path_folder,
                             f'test/{index_chunk[i]}/AIRS-CH0_signal.parquet'))
            signal = df.values.astype(np.float64).reshape(
                (df.shape[0], 32, 356))
            gain = test_adc_info['AIRS-CH0_adc_gain'].loc[index_chunk[i]]
            offset = test_adc_info['AIRS-CH0_adc_offset'].loc[index_chunk[i]]
            signal = ADC_convert(signal, gain, offset)
            dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
            dt_airs[1::2] += 0.1
            chopped_signal = signal[:, :, cut_inf:cut_sup]
            del signal, df

            # Clean AIRS data
            flat = pd.read_parquet(
                os.path.join(
                    path_folder,
                    f'test/{index_chunk[i]}/AIRS-CH0_calibration/flat.parquet')
            ).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
            dark = pd.read_parquet(
                os.path.join(
                    path_folder,
                    f'test/{index_chunk[i]}/AIRS-CH0_calibration/dark.parquet')
            ).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
            dead_airs = pd.read_parquet(
                os.path.join(
                    path_folder,
                    f'test/{index_chunk[i]}/AIRS-CH0_calibration/dead.parquet')
            ).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
            linear_corr = pd.read_parquet(
                os.path.join(
                    path_folder,
                    f'test/{index_chunk[i]}/AIRS-CH0_calibration/linear_corr.parquet'
                )).values.astype(np.float64).reshape(
                    (6, 32, 356))[:, :, cut_inf:cut_sup]

            if DO_MASK:
                chopped_signal = mask_hot_dead(chopped_signal, dead_airs, dark)
                AIRS_CH0_clean[i] = chopped_signal
            else:
                AIRS_CH0_clean[i] = chopped_signal

            if DO_THE_NL_CORR:
                linear_corr_signal = apply_linear_corr(linear_corr,
                                                       AIRS_CH0_clean[i])
                AIRS_CH0_clean[i, :, :, :] = linear_corr_signal
            del linear_corr

            if DO_DARK:
                cleaned_signal = clean_dark(AIRS_CH0_clean[i], dead_airs, dark,
                                            dt_airs)
                AIRS_CH0_clean[i] = cleaned_signal
            del dark

            # Load and process FGS1 signal
            df = pd.read_parquet(
                os.path.join(path_folder,
                             f'test/{index_chunk[i]}/FGS1_signal.parquet'))
            fgs_signal = df.values.astype(np.float64).reshape(
                (df.shape[0], 32, 32))

            FGS1_gain = test_adc_info['FGS1_adc_gain'].loc[index_chunk[i]]
            FGS1_offset = test_adc_info['FGS1_adc_offset'].loc[index_chunk[i]]

            fgs_signal = ADC_convert(fgs_signal, FGS1_gain, FGS1_offset)
            dt_fgs1 = np.ones(len(fgs_signal)) * 0.1
            dt_fgs1[1::2] += 0.1
            chopped_FGS1 = fgs_signal
            del fgs_signal, df

            # Clean FGS1 data
            flat = pd.read_parquet(
                os.path.join(
                    path_folder,
                    f'test/{index_chunk[i]}/FGS1_calibration/flat.parquet')
            ).values.astype(np.float64).reshape((32, 32))
            dark = pd.read_parquet(
                os.path.join(
                    path_folder,
                    f'test/{index_chunk[i]}/FGS1_calibration/dark.parquet')
            ).values.astype(np.float64).reshape((32, 32))
            dead_fgs1 = pd.read_parquet(
                os.path.join(
                    path_folder,
                    f'test/{index_chunk[i]}/FGS1_calibration/dead.parquet')
            ).values.astype(np.float64).reshape((32, 32))
            linear_corr = pd.read_parquet(
                os.path.join(
                    path_folder,
                    f'test/{index_chunk[i]}/FGS1_calibration/linear_corr.parquet'
                )).values.astype(np.float64).reshape((6, 32, 32))

            if DO_MASK:
                chopped_FGS1 = mask_hot_dead(chopped_FGS1, dead_fgs1, dark)
                FGS1_clean[i] = chopped_FGS1
            else:
                FGS1_clean[i] = chopped_FGS1

            if DO_THE_NL_CORR:
                linear_corr_signal = apply_linear_corr(linear_corr,
                                                       FGS1_clean[i])
                FGS1_clean[i, :, :, :] = linear_corr_signal
            del linear_corr

            if DO_DARK:
                cleaned_signal = clean_dark(FGS1_clean[i], dead_fgs1, dark,
                                            dt_fgs1)
                FGS1_clean[i] = cleaned_signal
            del dark

        # SAVE DATA AND FREE SPACE
        AIRS_cds = get_cds(AIRS_CH0_clean)
        FGS1_cds = get_cds(FGS1_clean)

        del AIRS_CH0_clean, FGS1_clean

        # (Optional) Time Binning
        if TIME_BINNING:
            AIRS_cds_binned = bin_obs(AIRS_cds, binning=30)
            FGS1_cds_binned = bin_obs(FGS1_cds, binning=30 * 12)
        else:
            AIRS_cds = AIRS_cds.transpose(0, 1, 3, 2)
            AIRS_cds_binned = AIRS_cds
            FGS1_cds = FGS1_cds.transpose(0, 1, 3, 2)
            FGS1_cds_binned = FGS1_cds

        del AIRS_cds, FGS1_cds

        for i in range(CHUNKS_SIZE):
            flat_airs = pd.read_parquet(
                os.path.join(
                    path_folder,
                    f'test/{index_chunk[i]}/AIRS-CH0_calibration/flat.parquet')
            ).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
            flat_fgs = pd.read_parquet(
                os.path.join(
                    path_folder,
                    f'test/{index_chunk[i]}/FGS1_calibration/flat.parquet')
            ).values.astype(np.float64).reshape((32, 32))
            if DO_FLAT:
                corrected_AIRS_cds_binned = correct_flat_field(
                    flat_airs, dead_airs, AIRS_cds_binned[i])
                AIRS_cds_binned[i] = corrected_AIRS_cds_binned
                corrected_FGS1_cds_binned = correct_flat_field(
                    flat_fgs, dead_fgs1, FGS1_cds_binned[i])
                FGS1_cds_binned[i] = corrected_FGS1_cds_binned

        # Save test data
        np.save(os.path.join(path_out, 'AIRS_clean_test_{}.npy'.format(n)),
                AIRS_cds_binned)
        np.save(os.path.join(path_out, 'FGS1_test_{}.npy'.format(n)),
                FGS1_cds_binned)
        del AIRS_cds_binned, FGS1_cds_binned
