import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import os


# 模型定义
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# 训练函数
def train_model(epochs=15, batch_size=128, lr=0.001):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    model = MNISTModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/mnist_model.pth')
    print("模型已保存到 models/mnist_model.pth")
    print("训练耗时: {:.3f} 秒".format(time.time() - start))


# 推理函数
def predict_image(image_path, model_path='models/mnist_model.pth'):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    result = []
    if os.path.isdir(image_path):
        for file in os.listdir(image_path):
            if file.endswith('.png'):
                image = Image.open(os.path.join(image_path, file))
                tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(tensor)
                    pred = output.argmax(dim=1).item()
                    result.append(pred)
                    print("file: {}, predict: {}".format(file, pred))
    else:
        print("image_path is not a directory")
        image = Image.open(image_path)
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1).item()
            result.append(pred)

        print("file: {}, predict: {}\n".format(image_path, pred))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST训练与推理脚本')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', type=str, help='预测图片路径')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')

    args = parser.parse_args()

    if args.train:
        train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.predict:
        predict_image(args.predict)
    else:
        print("请指定 --train 或 --predict 参数")