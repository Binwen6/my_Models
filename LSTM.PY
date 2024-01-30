import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST数据集的预处理和加载
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为RGB图像
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 使用预训练的ResNet作为特征提取器
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 替换分类层
model = model.to(device)

# 定义优化器和学习率调整器
optimizer = optim.AdamW(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 训练模型
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# 在验证集上评估模型
def evaluate(model, device, data_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
    val_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    return val_loss, accuracy

# 训练和验证模型
best_val_loss = float('inf')
for epoch in range(10):
    train(model, device, train_loader, optimizer, criterion)
    val_loss, accuracy = evaluate(model, device, test_loader, criterion)
    
    # 更新学习率
    scheduler.step(val_loss)
    
    # 保存具有最佳验证集性能的模型权重
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'resnet_best_model.pt')

    print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

# 加载具有最佳验证集性能的模型权重并在测试集上进行评估
model.load_state_dict(torch.load('resnet_best_model.pt'))
test_loss, test_accuracy = evaluate(model, device, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')