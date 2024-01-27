import xgboost as xgb

# 创建DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 参数设置
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'multi:softmax',
    'num_class': 3
}
epochs = 10

# 训练模型
model = xgb.train(params, dtrain, epochs)

# 预测与评估
predictions = model.predict(dtest)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
