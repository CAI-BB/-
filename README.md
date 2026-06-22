# 项目结构

```
mnist-cnn-pytorch/
│
├── demo1.py                  # Baseline model (2 Conv Layers): training, evaluation, inference
├── demo2.py                  # Original reference code (https://www.ctyun.cn/developer/article/710730097610821)
├── demo3.py                  # Extended model (3 Conv Layers): training, evaluation, inference
│
├── models/                   # Model save directory (auto-generated after training)
│   └── mnist_model.pth       # Trained model weights file
│
├── data/                     # MNIST dataset directory (auto-downloaded on first run)
│   └── MNIST/                # Raw image data
│
├── picture/                  # Custom prediction image directory (user-created)
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
│
└── README.md                 # Project documentation and setup guide
```
# 安装
## 1.Miniconda
安装教程：https://www.cnblogs.com/ajianbeyourself/p/17310681.html
## 2.Visual Studio Code

# 运行步骤
## 命令行运行
打开终端，进入项目根目录后执行：
### 运行命令

| 操作 | 命令 | 说明 |
|---|---|---|
| 训练基准模型 | ` demo1.py --train` | 使用2层卷积CNN训练15轮，自动保存模型至 `models/` |
| 评估已有模型 | ` demo1.py --eval` | 加载已训练模型，输出测试集准确率 |
| 单张图片预测 | ` demo1.py --predict test.png` | 预测单张图片（支持 png/jpg/jpeg/bmp/tiff） |
| 批量图片预测 | ` demo1.py --predict picture/` | 预测 `picture/` 文件夹内所有图片 |


### 可选高级参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | 15 | 训练轮数，可增大以提升精度 |
| `--batch_size` | 128 | 批次大小，可根据显存/内存调整 |
| `--lr` | 0.001 | 学习率，AdamW优化器的默认推荐值 |
| `--model_path` | `models/mnist_model.pth` | 模型文件加载/保存路径 |
