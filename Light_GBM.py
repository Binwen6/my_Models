import lightgbm as lgb

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
