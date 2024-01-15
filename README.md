# FIN



## [FIN: Flow-based Robust Watermarking with Invertible Noise Layer for Black-box Distortions](https://ojs.aaai.org/index.php/AAAI/article/view/25633)



Han Fang, Yupeng Qiu, Kejiang Chen*, Jiyi Zhang, Weiming Zhang, and Ee-Chien Chang*.

> 原文章：Flow-based Robust Watermarking with Invertible Noise Layer for Black-box Distortions, which is received by AAAI' 23.
> 本仓库是在此基础之上进行了一些改进
> 

****

### Requirements

The core packages we use in the project and their version information are as follows:

- kornia `0.6.6`
- natsort `7.1.1`
- numpy `1.22.3`
- pandas `1.4.3`
- torch `1.12.0`
- torchvision `0.13.0`

****

### Dataset

In this project we use DIV2K as the training dataset(which contains 800 images) and the validing dataset (which contians 100 images).

The data directory has the following structure:
```
├── data
│   ├── DIV2K_train
│   │   ├── 0001.png
│   │   ├── ...
│   ├── DIV2K_valid
│   │   ├── 0801.png
│   │   ├── ...
├── 

```


****


## 复现改进方案的过程
### 步骤1：训练模型
```bash
python train.py
```

### 步骤2：测试
```
├── experiments
│   ├── JPET
│   │   ├── FED.pt
│   │   ├── ...
│   ├── my.py
│   ├── reproduction.pt
│   ├── ...
├── 

```
这里已经有已经训练好的三个主要的模型参数文件，分别是`FED.pt`（是作者提供的）、
`reproduction.pt`（在我自己的实验环境中复现出来的）和`my.py`（我改进的）。

#### 水印嵌入
执行以下代码对16张测试图像进行水印嵌入：
```bash
# MODEL_NAME可以是author、re、my，分别表示使用以上三个模型进行水印嵌入
python encode.py --testing-model $MODEL_NAME$  
```

#### 提取水印
执行以下代码提取16张已经嵌入水印的图片中的水印信息：
```bash
# MODEL_NAME可以是author、re、my，分别表示使用以上三个模型进行水印嵌入
python decode.py --testing-model $MODEL_NAME$  
```
执行之前可先对`config.py`文件中的noises参数进行设置，选择所需噪声层和对应参数。
> 噪声层代码来自MBRS: Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression

执行decode时会先将已嵌入水印的图片进行加噪，然后再提取水印，并计算所提取水印的错误率。

