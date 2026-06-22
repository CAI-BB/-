import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import os


# 模型定义（保持不变）
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


# 训练函数（增加测试集评估）
def train_model(epochs=15, batch_size=128, lr=0.001):
 if torch.cuda.is_available():
    device = torch.device("cuda")
 elif torch.mps.is_available():
    device = torch.device("mps")
 else:
    device = torch.device("cpu")
    print(f"使用设备: {device}")

    model = MNISTModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载训练集和测试集
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

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
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')

    # 训练结束后在测试集上评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total
    print(f'测试集准确率: {acc:.2f}%')

    # 保存模型
    os.makedirs('models', exist_ok=True)
    model_path = 'models/mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")
    print("训练耗时: {:.3f} 秒".format(time.time() - start))


# 仅评估测试集（加载已有模型）
def evaluate_model(model_path='models/mnist_model.pth'):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在")
        return
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total
    print(f'测试集准确率: {acc:.2f}%')


# 推理函数（支持多种图片格式）
def predict_image(image_path, model_path='models/mnist_model.pth'):
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在，请先训练或指定正确路径")
        return

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

    # 支持的图片扩展名
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    if os.path.isdir(image_path):
        print(f"预测文件夹: {image_path}")
        for file in os.listdir(image_path):
            if file.lower().endswith(valid_exts):
                img_path = os.path.join(image_path, file)
                image = Image.open(img_path)
                tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(tensor)
                    pred = output.argmax(dim=1).item()
                    # 获取置信度（可选）
                    prob = torch.softmax(output, dim=1).max().item()
                print(f"file: {file:20s} -> predict: {pred}  (confidence: {prob:.3f})")
    else:
        print("预测单个文件")
        image = Image.open(image_path)
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1).item()
            prob = torch.softmax(output, dim=1).max().item()
        print(f"file: {os.path.basename(image_path)} -> predict: {pred}  (confidence: {prob:.3f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST训练、评估与推理脚本')
    parser.add_argument('--train', action='store_true', help='训练模型（自动评估测试集）')
    parser.add_argument('--eval', action='store_true', help='仅评估测试集（需已有模型）')
    parser.add_argument('--predict', type=str, help='预测图片路径（文件或文件夹）')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--model_path', type=str, default='models/mnist_model.pth', help='模型文件路径')

    args = parser.parse_args()

    if args.train:
        train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.eval:
        evaluate_model(model_path=args.model_path)
    elif args.predict:
        predict_image(args.predict, model_path=args.model_path)
    else:
        print("请指定 --train、--eval 或 --predict 参数")
        print("示例：")
        print("  训练: & C:/ /miniconda3/python.exe c:/ /Desktop/  /demo1.py --train ")
        print("  评估:  & C:/ /miniconda3/python.exe c:/ /Desktop/  /demo1.py --eval")
        print("  预测: & C:/ /miniconda3/python.exe c:/ /Desktop/  /demo1.py --predict picture")
        print("  指定模型:& C:/ /miniconda3/python.exe c:/ /Desktop/  /demo1.py --predict picture --model_path models/")