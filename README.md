项目结构
mnist-cnn-pytorch/
│
├── demo1.py                  # Baseline model (2 Conv Layers): training, evaluation, inference
├── demo2.py                  # Original reference code (training + single image inference only)
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
├── requirements.txt          # Python dependencies list
└── README.md                 # Project documentation and setup guide
