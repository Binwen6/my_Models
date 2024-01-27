# Perfect model for this dataset
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 创建数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 参数设置
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss'
}

# 训练模型
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

# 预测与评估
y_pred = model.predict(X_test)
y_pred_max = [np.argmax(line) for line in y_pred]
accuracy = accuracy_score(y_test, y_pred_max)
print(f"Accuracy: {accuracy}")
