import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST数据集的预处理和加载
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# 划分数据集
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义LSTM模型
class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

model = LSTMNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train(model, device, train_loader, optimizer, criterion, epochs):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = data.squeeze().view(-1, 28, 28)  # 调整数据形状以适应LSTM输入
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

        # 在验证集上评估模型性能
        val_loss = evaluate(model, device, val_loader, criterion)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')

        # 保存具有最佳验证集性能的模型权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'lstm_best_model.pt')

# 在验证集上评估模型
def evaluate(model, device, data_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            data = data.squeeze().view(-1, 28, 28)  # 调整数据形状以适应LSTM输入
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)
    val_loss /= len(data_loader.dataset)
    return val_loss

# 测试模型
def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.squeeze().view(-1, 28, 28)  # 调整数据形状以适应LSTM输入
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f'Test set: Accuracy: {100. * correct / len(test_loader.dataset)}%')

# 执行训练和测试
epochs = 3  # 设置训练的轮数
train(model, device, train_loader, optimizer, criterion, epochs)
test(model, device, test_loader)