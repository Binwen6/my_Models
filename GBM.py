from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载波士顿房价数据集
boston = load_boston()
X, y = boston.data, boston.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)

# 梯度提升机模型
gbm = GradientBoostingRegressor(n_estimators=100, random_state=0)
gbm.fit(X_train, y_train)
gbm_pred = gbm.predict(X_test)
gbm_mse = mean_squared_error(y_test, gbm_pred)

rf_mse, gbm_mse  # 输出两个模型的均方误差(MSE)进行比较

