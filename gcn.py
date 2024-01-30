import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
import torch
# 加载Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora')

class EnhancedGCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(EnhancedGCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, dataset.num_classes, heads=2, dropout=0.6)
        self.dropout = 0.5

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

model = EnhancedGCN(hidden_channels=16)
data = dataset[0]

# 划分数据集
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

# 训练模型
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=4e-2)
best_val_loss = float('inf')
best_model = None

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    # 验证损失
    model.eval()
    with torch.no_grad():
        val_out = model(data.x, data.edge_index)
        val_loss = F.nll_loss(val_out[val_mask], data.y[val_mask])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    model.train()

    if epoch % 10 == 0:
        print(f'Epoch {epoch} | Loss: {loss.item()} | Val Loss: {val_loss.item()}')

# 保存最佳模型
torch.save(best_model, 'gcn_best_model.pt')

# 测试模型
model.load_state_dict(best_model)
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
correct = int(pred[test_mask].eq(data.y[test_mask]).sum().item())
accuracy = correct / int(test_mask.sum())
print(f'Test Accuracy: {accuracy}')