import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.datasets import co2

# 加载CO2数据集作为时间序列示例
data = co2.load_pandas().data
data = data['co2'].resample('MS').mean()
data = data.fillna(data.bfill())

# 取最后12个数据点作为测试集
train = data[:-12]
test = data[-12:]

# 定义ARIMA模型并拟合
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# 进行预测
forecast = model_fit.forecast(steps=12)

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast, label='Forecast')
plt.title('CO2 Levels Forecast')
plt.legend()
plt.show()
