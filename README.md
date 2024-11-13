## English

### utility .py

- Competition scoring, competition upload file processing, feature engineering, training, prediction function

### linear_process.py

- Multi-process preprocessing of raw data, output shape is [planet, time, wavelength]
- This is mainly used for later models

### preprocessing.py

- Single-process raw data preprocessing: output shape is [planet, time, wavelength, space]
- This is used for early models

### Features:

White light scaling ratio S: S for each planet

Relative area of ​​absorption peaks for each band of each planet: Alpha

Luminous flux map for each band of each planet (compressed space, only 3 dimensions): FLUX

Luminous flux map for each band of each planet (uncompressed space, 4 dimensions): SFLUX

### Training and model

- Each file in model_train is the model code and the corresponding training code

### Prediction

- ST_Predict.ipynb and STws_Predict.ipynb is the prediction code of the two models
- calsigma.ipynb is used for sigma calculation of ST model

### Model files and preprocessed raw data are shown in the following link

https://drive.google.com/drive/folders/1TXQ_3aluSlxsepPtaJtTAPKqHCF2APcG?usp=sharing

- train_preprocessed.pkl is the preprocessed data
- best_model_STws_v3.5.pth and best_model_ST_v3.3.pth are the trained models

### Raw data

https://www.kaggle.com/competitions/ariel-data-challenge-2024/data

## 中文

### utility .py

- 比赛计分，比赛上传文件处理，特征工程，训练，预测函数

### linear_process.py

- 多进程预处理原始数据，输出形状是[行星，时间，波长]
- 后期模型主要用这个

### preprocessing.py

- 单进程原始数据预处理：输出形状是[行星，时间，波长，空间]
- 前期模型用这个

### 特征：

每个星球的白光缩放比例 S：S

每个星球的各个波段的吸收峰相对面积：Alpha

每个星球的各个波段的光通量图（已压缩空间，只有 3 个维度）: FLUX

每个星球的各个波段的光通量图（未压缩空间，有 4 个维度）: SFLUX

### 训练及模型

- model_train 里每一个文件都是模型代码和对应的训练代码

### 预测

- ST_Predict.ipynb 和 STws_Predict.ipynb 分别是两个模型的预测代码
- calsigma.ipynb 则是用于 ST 模型的 sigma 计算

### 模型文件和预处理好的原始数据见以下链接

https://drive.google.com/drive/folders/1TXQ_3aluSlxsepPtaJtTAPKqHCF2APcG?usp=sharing

- train_preprocessed.pkl 是预处理好的数据
- best_model_STws_v3.5.pth 和 best_model_ST_v3.3.pth 则是训练好的模型

### 原始数据

https://www.kaggle.com/competitions/ariel-data-challenge-2024/data
