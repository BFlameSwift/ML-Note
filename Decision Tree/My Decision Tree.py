from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate  # 划分数据集函数
from sklearn.metrics import accuracy_score  # 准确率函数
import math
# import cvxpy
# import torch

RANDOM_SEED = 2020 # 固定随机种子


# 读入数据

csv_data = './data/high_diamond_ranked_10min.csv' # 数据路径
data_df = pd.read_csv(csv_data, sep=',') # 读入csv文件为pandas的DataFrame
data_df = data_df.drop(columns='gameId') # 舍去对局标号列



print(data_df.iloc[0]) # 输出第一行数据
data_df.describe() # 每列特征的简单统计信息


# 增删特征

drop_features = ['blueGoldDiff', 'redGoldDiff', 
                 'blueExperienceDiff', 'redExperienceDiff', 
                 'blueCSPerMin', 'redCSPerMin', 
                 'blueGoldPerMin', 'redGoldPerMin'] # 需要舍去的特征列
df = data_df.drop(columns=drop_features) # 舍去特征列
info_names = [c[3:] for c in df.columns if c.startswith('red')] # 取出要作差值的特征名字（除去red前缀）
for info in info_names: # 对于每个特征名字
    df['br' + info] = df['blue' + info] - df['red' + info] # 构造一个新的特征，由蓝色特征减去红色特征，前缀为br
# 其中FirstBlood为首次击杀最多有一只队伍能获得，brFirstBlood=1为蓝，0为没有产生，-1为红
df = df.drop(columns=['blueFirstBlood', 'redFirstBlood']) # 原有的FirstBlood可删除


# 特征离散化
# 决策树ID3算法一般是基于离散特征的，本例中存在很多连续的数值特征，做离散化实现。

not_process_num = 7

discrete_df = df.copy() # 先复制一份数据
for c in df.columns[1:]: # 遍历每一列特征，跳过标签列
    if len(df[c].unique()) <= not_process_num:
        continue
    else:
        discrete_df[c] = pd.qcut(df[c], not_process_num, precision=0, labels=False, duplicates='drop')


all_y = discrete_df['blueWins'].values # 所有标签数据
feature_names = discrete_df.columns[1:] # 所有特征的名称
all_x = discrete_df[feature_names].values # 所有原始特征值

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)



# 定义决策树类
class DecisionTree(object):
    def __init__(self, classes, features, 
                 max_depth=10, min_samples_split=10,
                 impurity_t='entropy'):


        # 传入一些可能用到的模型参数，也可能不会用到
        # classes表示模型分类总共有几类
        # features是每个特征的名字，也方便查询总的共特征数
        # max_depth表示构建决策树时的最大深度
        # min_samples_split表示构建决策树分裂节点时，如果到达该节点的样本数小于该值则不再分裂
        # impurity_t表示计算混杂度（不纯度）的计算方式，例如entropy或gini

        self.classes = classes
        self.features = features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_t = impurity_t
        self.root = None # 定义根节点，未训练时为空
        

    def impurity(self,data):
        cnt = Counter(data)
        proportion = [cnt[i]/len(data) for i in cnt] # 计算每一个数值所占data的比例
        # print("cnt=",cnt)
        # print("proportion",proportion)
        if self.impurity_t == 'entropy': # 如果需要信息熵
            ret = 0
            for i in range(len(cnt)):
                if(proportion[i]>0):
                    ret -= proportion[i]*math.log2(proportion[i])
            # print(ret)
            return ret,cnt      # 信息熵：entropy = -\sum{i=1,n} (pi·logpi)
        elif self.impurity_t == 'gini':
            ret = 0
            for i in range(len(cnt)):
                ret += proportion[i]*proportion[i]
            # print('gini',1-ret)
            return (1-ret),cnt  # gini系数：Gini = 1-∑(k=1,n)|(Pk)^2
        else:
            print('error, impurity错误')
    def gain(self, feature, label):
        c_impurity, _ = self.impurity(label)  # 不考虑特征时标签的混杂度

        # 记录特征的每种取值所对应的数组下标
        f_index = {}
        for idx, v in enumerate(feature):
            if v not in f_index:
                f_index[v] = []
            f_index[v].append(idx)

        # 计算根据该特征分裂后的不纯度，根据特征的每种值的数目加权和
        f_impurity = 0
        for v in f_index:
            f_l = label[f_index[v]]  # 取出该特征取值对应的数组下标
            f_impurity += self.impurity(f_l)[0] * len(f_l) / len(label)  # 计算不纯度并乘以该特征取值的比例


        r = self.impurity(feature)[0] # 计算该特征在标签无关时的不纯度
        r = (c_impurity - f_impurity)/r if r > 0 else c_impurity - f_impurity # 除数不为0时为信息增益率
        return r, f_index # 返回信息增益率，以及每个特征取值的数组下标，方便之后使用

    def expand_node(self, feature, label, depth, skip_features=set()):
        # 分裂节点，feature和label为到达该节点的样本
        # feature为二维numpy（n*m）数组，每行表示一个样本，有m个特征
        # label为一维numpy（n）数组，表示每个样本的分类标签
        # depth记录了当前节点的深度
        # skip_features表示当前路径已经用到的特征
        # 在当前ID3算法离散特征的实现下，一条路径上已经用过的特征不会再用（其他实现有可能会选重复特征）
        l_cnt = Counter(label)  # 计数每个类别的样本出现次数
        if len(l_cnt) <= 1:  # 如果只有一种类别了，无需分裂，已经是叶节点
            return label[0]  # 只需记录类别
        if len(label) < self.min_samples_split or depth > self.max_depth:  # 如果达到了最小分裂的样本数或者最大深度的阈值
            return l_cnt.most_common(1)[0][0]  # 则只记录当前样本中最多的类别

        f_idx, max_gain, fv_index = -1, -1, None  # 准备挑选分裂特征
        for idx in range(len(self.features)):  # 遍历所有特征
            if idx in skip_features:  # 如果当前路径已经用到，不用再算
                continue
            f_gain, fv = self.gain(feature[:, idx], label)  # 计算特征的信息增益，fv是特征每个取值的样本下标

            if f_gain <= 0: # 如果信息增益不为正，跳过该特征
               continue
            if f_idx < 0 or f_gain > max_gain:  # 如果个更好的分裂特征
                f_idx, max_gain, fv_index = idx, f_gain, fv  # 则记录该特征

        if f_idx < 0:  # 如果没有找到合适的特征，即所有特征都没有信息增益
            return l_cnt.most_common(1)[0][0]  # 则只记录当前样本中最多的类别

        decision = {}  # 用字典记录每个特征取值所对应的子节点，key是特征取值，value是子节点
        skip_features = set([f_idx] + [f for f in skip_features])  # 子节点要跳过的特征包括当前选择的特征
        for v in fv_index:  # 遍历特征的每种取值
            decision[v] = self.expand_node(feature[fv_index[v], :], label[fv_index[v]],  # 取出该特征取值所对应的样本
                                           depth=depth + 1, skip_features=skip_features)  # 深度+1，递归调用节点分裂
        # 返回一个元组，有三个元素
        # 第一个是选择的特征下标，第二个特征取值和对应的子节点，第三个是到达当前节点的样本中最多的类别
        return (f_idx, decision, l_cnt.most_common(1)[0][0])

    def traverse_node(self, node, feature):
        # 预测样本时从根节点开始遍历节点，根据特征路由。
        # node表示当前到达的节点，例如self.root
        # feature是长度为m的numpy一维数组
        assert len(self.features) == len(feature)  # 要求输入样本特征数和模型定义时特征数目一致
        if type(node) is not tuple:  # 如果到达了一个节点是叶节点（不再分裂），则返回该节点类别
            return node
        fv = feature[node[0]]  # 否则取出该节点对应的特征值，node[0]记录了特征的下标
        if fv in node[1]:  # 根据特征值找到子节点，注意需要判断训练节点分裂时到达该节点的样本是否有该特征值（分支）
            return self.traverse_node(node[1][fv], feature)  # 如果有，则进入到子节点继续遍历
        return node[-1]  # 如果没有，返回训练时到达当前节点的样本中最多的类别

    def fit(self, feature, label):

        assert len(self.features) == len(feature[0])
        self.root = self.expand_node(feature, label, depth=1)

    def predict(self, feature):

        assert len(feature.shape) == 1 or len(feature.shape) == 2  # 只能是1维或2维
        if len(feature.shape) == 1:  # 如果是一个样本
            return self.traverse_node(self.root, feature)  # 从根节点开始路由
        return np.array([self.traverse_node(self.root, f) for f in feature])  # 如果是很多个样本

    def get_params(self, deep):

        return {'classes': self.classes, 'features': self.features,
                'max_depth': self.max_depth, 'min_samples_split': self.min_samples_split,
                'impurity_t': self.impurity_t}


    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        

DT = DecisionTree(classes=[0,1], features=feature_names, max_depth=5, min_samples_split=10, impurity_t='entropy')

DT.fit(x_train, y_train) # 在训练集上训练
p_test = DT.predict(x_test) # 在测试集上预测，获得预测值
print(p_test) # 输出预测值
test_acc = accuracy_score(p_test, y_test) # 将测试预测值与测试集标签对比获得准确率
print('accuracy: {:.4f}'.format(test_acc)) # 输出准确率
# accuracy = 0.7131


