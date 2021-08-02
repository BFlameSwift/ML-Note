#!/usr/bin/env python
# coding: utf-8

# 基于回归分析的大学综合得分预测

# 使用来自 Kaggle 的[数据](https://www.kaggle.com/mylesoneill/world-university-rankings?select=cwurData.csv)，构建「线性回归」模型，根据大学各项指标的排名预测综合得分。
# In[372]:


import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


csv_data = "./cwurData.csv"
raw_data = pd.read_csv(csv_data,sep=",")


# 去除其中包含 NaN 的数据，保留 2000 条有效记录。

# In[373]:


raw_data = raw_data.dropna()
len(raw_data)


# 取出对应自变量以及因变量的列，之后就可以基于此切分训练集和测试集，并进行模型构建与分析。
# 将数据转化为字典形式进行处理

# In[374]:


coefficient = {}
coefficient['education'] = []; coefficient['employment'] = [];coefficient['faculty'] = [];coefficient['publications'] = []
coefficient['influence'] = [];coefficient['citations'] = [];coefficient['broad_impact'] = []; coefficient['patents'] = [];coefficient['school_name'] = []


# ## 四、模型构建

# 

# In[375]:


data = np.array(raw_data)
np.random.shuffle(data)
num = 0
print("自变量数目为：",len(raw_data.columns))

coefficient['education'] = np.array(data[:,4:5:1])

attitude_colums  = ['education','employment','faculty','publications','influence','citations','broad_impact','patents']

# for i in range(len(data)):
for j in range(5,13,1):
    coefficient["{}".format(attitude_colums[j-5])] = np.concatenate(np.array(data[:,j-1:j:1]))
score = np.concatenate(np.array(data[:,12:13]))


# 划分测试集与训练集
all_y = data[:,12]
all_x = data[:,4:12]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(all_x,all_y,test_size=0.2,random_state=2021)


md = LinearRegression().fit(x_train,y_train)
y_predict = md.predict(x_test)
b0 = md.intercept_
b1_8 = md.coef_
R2 = md.score(x_test,y_test)


test_bias = y_predict-y_test
test_rmse = math.sqrt(sum([i**2 for i in test_bias])/len(test_bias))


# 输出相关系数$\beta$、拟合优度与RMSE 

# In[376]:


print("相关系数beta ",b0,",".join(str(i) for i in b1_8))
print("拟合优度 = ",R2)
print("RMSE = ",test_rmse)


# ## 岭回归求解

# 通过拟合优度的计算与RMSE的计算发现拟合效果一般，可知拟合效果较差与自变量间的共线性有关，于是对自变量进行正则化处理，进而使用岭回归进行求解

# 导入岭回归所需库，以及创建一些参数

# In[377]:


import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,RidgeCV
from scipy.stats import zscore

b_ridge = []


# 调库对数据进行正则化处理得到标准化回归系数，而后求出岭回归的回归系数

# In[378]:


kk = np.logspace(-4,10,100)
for k in kk:
    md_ridge = Ridge(alpha=k).fit(x_train,y_train)
    b_ridge.append(md_ridge.coef_)

md_ridge_cv = RidgeCV(alphas=np.logspace(-4,10,100)).fit(x_train,y_train)
print("最优alpha = ",md_ridge_cv.alpha_)

md_ridge_0 = Ridge(0.4).fit(x_test,y_test)
cs0 = md_ridge_0.coef_
print("标准化数据的回归系数为：",cs0)
mu=np.mean(data[:,4:13],axis=0); 
print(mu)
s=np.std(data[:,4:13],dtype=np.float64, axis=0,ddof=1) #计算所有指标的均值和标准差
params=[mu[-1]-s[-1]*sum(cs0*mu[:-1]/s[:-1]),s[-1]*cs0/s[:-1]]
print("原数据的回归系数为：",params)


# 在测试集上评估岭回归训练数据

# In[379]:


y_predict_ridge = md_ridge_0.predict(x_test)
test_bias_ridge = y_predict_ridge-y_test
test_rmse_ridge = math.sqrt(sum([i**2 for i in test_bias_ridge])/len(test_bias_ridge))




print("处理后的拟合优度：",md_ridge_0.score(x_test,y_test),">处理前拟合优度",R2)
print("相关系数beta: ",",".join(str(i) for i in params))
print("处理后RMSE = ",test_rmse_ridge,"<处理前RMSE = ",test_rmse)
#处理后RMSE =  2.1163977771613025 <处理前RMSE =  2.8259342982745057


# 使用岭回归对数据做拟合后发现拟合效果优于直接对数据进行线性拟合
