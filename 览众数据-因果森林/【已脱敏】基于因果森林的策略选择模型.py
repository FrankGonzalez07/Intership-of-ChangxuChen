#!/usr/bin/env python
# coding: utf-8

# # 调包&读数

# In[126]:


# Some imports to get us started
# Utilities
import os
import urllib.request
import numpy as np
import pandas as pd

# Generic ML imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor

# EconML imports
from econml.dml import LinearDML, CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter, SingleTreePolicyInterpreter

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# In[4]:


# Import the sample pricing data
file_url = "https://msalicedatapublic.blob.core.windows.net/datasets/Pricing/pricing_sample.csv"
train_data = pd.read_csv(file_url)
train_data


# # 构建相关变量

# In[5]:


train_data.nunique()


# In[6]:


# Define estimator inputs
Y = train_data["demand"]  # outcome of interest
T = train_data["price"]  # intervention, or treatment
X = train_data[["income"]]  # features
W = train_data.drop(columns=["demand", "price", "income"])  # confounders


# # 模拟仿真数据

# In[9]:


# Get test data
X_test = np.linspace(0, 5, 100).reshape(-1, 1)
X_test_data = pd.DataFrame(X_test, columns=["income"])


# In[12]:


X_test_data


# # 构建DGP生成处理效应及其上下界

# In[55]:


# Define underlying treatment effect function given DGP
def gamma_fn(X):
    return -3 - 14 * (X["income"] < 1)   
    # - 14 * (X["income"] < 1)：如果不符合条件，则化为0，再去跟-3相加；如果符合条件，则化为1，再去相乘
    # 此处之所以让仿真数据的价格出现跳跃，就是为了测试模型的拟合效果（实际效果看拟合图部分）

def beta_fn(X):
    return 20 + 0.5 * (X["avg_hours"]) + 5 * (X["days_visited"] > 4)

def true_te(x, n, stats):
    if x < 1:
        subdata = train_data[train_data["income"] < 1].sample(n=n, replace=True)
        # 只取 income 小于1的样本，并有放回地抽样 n 个上来
    else:
        subdata = train_data[train_data["income"] >= 1].sample(n=n, replace=True)
        # 只取 income 大于等于1的样本，并有放回地抽样 n 个上来
    # 第一个 if 和 else 中的条件并无特殊意义，只是为了打乱随机之用
    
    te_array = subdata["price"] * gamma_fn(subdata) / (subdata["demand"])
    if stats == "mean":
        return np.mean(te_array)  # 按照输入参数指令，计算出一维数组的均值
    elif stats == "median":
        return np.median(te_array)# 按照输入参数指令，计算出一维数组的中位数
    elif isinstance(stats, int):  # 如果参数指令既不是"mean"也不是"median"，
                                  # 用 isinstance 函数判断传入的参数是否是整数，如果是，将 te_array 排序后求取相应的百分位数
        return np.percentile(te_array, stats)


# In[112]:


subdata.sort_values(by = ['income'])


# In[104]:


gamma_fn(subdata)


# In[101]:


subdata = train_data[train_data["income"] >= 1].sample(n=1000, replace=True)


# In[102]:


te_array = subdata["price"] * gamma_fn(subdata) / (subdata["demand"])


# In[103]:


te_array


# In[93]:


truth_te_estimate = np.apply_along_axis(true_te, 1, X_test, 1000, "mean")  
# true_te 的作用详解：逐个遍历 X_test 的元素，
# 如果 X_test 中的某个元素小于1，则构建相应的 subdata，并根据 subdata 构建出相应的 te_array（是个array），
# 再根据传入的参数，求出这个 te_array 的均值、中位数或百分位数

# 这里之所以 axis 参数为1是因为 X_test 中每个数值都包含在一个"[]"中，即每个数值都是单独一列，如果所有数值都在一个”[]“中，则 axis 参数为0
# 1000, "mean" 都是apply_along_axis的附加参数，都来自于 func（即 true_te），可有可无

truth_te_upper = np.apply_along_axis(true_te, 1, X_test, 1000, 95)  # 以95%分位数为上界
truth_te_lower = np.apply_along_axis(true_te, 1, X_test, 1000, 5)  # 以5%分位数为下界


# In[98]:


truth_te_estimate


# In[95]:


truth_te_upper


# In[96]:


truth_te_lower


# # 参数异质性

# In[113]:


# Get log_T and log_Y
log_T = np.log(T)
log_Y = np.log(Y)


# In[120]:


log_T


# In[121]:


log_Y


# In[114]:


# 因为捕捉的是参数异质性，因而此处使用LinearDML估计器
est = LinearDML(
    model_y = GradientBoostingRegressor(),
    model_t = GradientBoostingRegressor(),
    featurizer = PolynomialFeatures(degree = 2, include_bias = False),
)
est.fit(log_Y, log_T, X = X, W = W, inference = "statsmodels")
# Get treatment effect and its confidence interval
te_pred = est.effect(X_test)
te_pred_interval = est.effect_interval(X_test)


# In[124]:


X_test
# 0到5的100等分等差数列


# In[122]:


te_pred


# In[123]:


te_pred_interval


# In[128]:


# Compare the estimate and the truth
plt.figure(figsize=(10, 6))
plt.plot(X_test.flatten(), te_pred, label="仿真价格弹性")
# X_test.flatten() 表示将分散的'[]'压成一个'[]'

plt.plot(X_test.flatten(), truth_te_estimate, "--", label="真实拟合的价格弹性")
plt.fill_between(
    X_test.flatten(),
    te_pred_interval[0],
    te_pred_interval[1],
    alpha=0.2,
    label="95% 的置信区间",
)
plt.fill_between(
    X_test.flatten(),
    truth_te_lower,
    truth_te_upper,
    alpha=0.2,
    label="真实拟合价格弹性的区间",
)
plt.xlabel("Income")
plt.ylabel("Songs Sales Elasticity")
plt.title("Songs Sales Elasticity vs Income")
plt.legend(loc="lower right")


# In[133]:


# 此处之所以让仿真数据的价格出现跳跃，就是为了测试模型的拟合效果


# In[118]:


X_test


# In[119]:


X_test.flatten()


# In[129]:


# Get the final coefficient and intercept summary
est.summary()


# In[147]:


# 从上图可以看出拟合的效果并不好，因而LinearDML估计器并不合适


# # 非参数异质性

# In[137]:


# Train EconML model
est_NonPara = CausalForestDML(
    model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor()
)
est_NonPara.fit(log_Y, log_T, X=X, W=W, inference="blb")  # blb：Bootstrap-of-Little-Bags based inference
# 注意：这个模型是基于真实的数据拟合而成，然后再基于仿真数据 X_test 进行预测

# Get treatment effect and its confidence interval
te_pred = est_NonPara.effect(X_test)
te_pred_interval = est_NonPara.effect_interval(X_test)


# In[138]:


# Compare the estimate and the truth
plt.figure(figsize=(10, 6))
plt.plot(X_test.flatten(), te_pred, label="仿真价格弹性")
plt.plot(X_test.flatten(), truth_te_estimate, "--", label="真实拟合的价格弹性")
plt.fill_between(
    X_test.flatten(),
    te_pred_interval[0],
    te_pred_interval[1],
    alpha=0.2,
    label="95% 的置信区间",
)
plt.fill_between(
    X_test.flatten(),
    truth_te_lower,
    truth_te_upper,
    alpha=0.2,
    label="真实拟合价格弹性的区间",
)
plt.xlabel("Income")
plt.ylabel("Songs Sales Elasticity")
plt.title("Songs Sales Elasticity vs Income")
plt.legend(loc="lower right")


# # 基于非参数异质性的条件平均处理效应分析

# In[146]:


X_test


# In[136]:


intrp = SingleTreeCateInterpreter(include_model_uncertainty = True, max_depth = 3, min_samples_leaf = 10)
intrp.interpret(est_NonPara, X_test)
plt.figure(figsize=(25, 5))
intrp.plot(feature_names=X.columns, fontsize=12)


# In[150]:


intrp = SingleTreePolicyInterpreter(risk_level=0.05, max_depth=2, min_samples_leaf=1, min_impurity_decrease=0.001)
intrp.interpret(est_NonPara, X_test, sample_treatment_costs= -1)
plt.figure(figsize=(25, 5))
intrp.plot(feature_names = X.columns, treatment_names=["Discount", "No-Discount"], fontsize=18)


# In[151]:


# 上图的含义：
# 对于收入小于0.985的群体来说，如果给予其discount，其价格下降最明显，说明该群体对discount的相应最显著


# # 方案利润评估

# In[ ]:


# Define estimator inputs
Y = train_data["demand"]  # outcome of interest
T = train_data["price"]  # intervention, or treatment
X = train_data[["income"]]  # features
W = train_data.drop(columns=["demand", "price", "income"])  # confounders


# In[153]:


def gamma_fn(X):
    return -3 - 14 * (X["income"] < 1)   
    # - 14 * (X["income"] < 1)：如果不符合条件，则化为0，再去跟-3相加；如果符合条件，则化为1，再去相乘
    # 此处之所以让仿真数据的价格出现跳跃，就是为了测试模型的拟合效果（实际效果看拟合图部分）

def beta_fn(X):
    return 20 + 0.5 * (X["avg_hours"]) + 5 * (X["days_visited"] > 4)

# define function to compute revenue
def demand_fn(data, T):
    Y = gamma_fn(data) * T + beta_fn(data)
    return Y

def revenue_fn(data, discount_level1, discount_level2, baseline_T, policy):
    policy_price = baseline_T * (1 - discount_level1) * policy + baseline_T * (1 - discount_level2) * (1 - policy)
    demand = demand_fn(data, policy_price)
    rev = demand * policy_price
    return rev


# In[158]:


policy = intrp.treat(X)
policy
# 这个 policy 是非参数模型 est_NonPara 建议的方案


# In[159]:


policy_dic = {}
# our policy above

policy = intrp.treat(X)
policy_dic["Our Policy"] = np.mean(revenue_fn(train_data, 0, 0.1, 1, policy))
# 非参数模型 est_NonPara 建议的方案

## previous strategy
policy_dic["Previous Strategy"] = np.mean(train_data["price"] * train_data["demand"])

## give everyone discount
policy_dic["Give Everyone Discount"] = np.mean(revenue_fn(train_data, 0.1, 0, 1, np.ones(len(X))))
# 给所有人都打折，为什么这么表示：观察”baseline_T * (1 - discount_level2) * (1 - policy)“

## don't give discount
policy_dic["Give No One Discount"] = np.mean(revenue_fn(train_data, 0, 0.1, 1, np.ones(len(X))))
# 给所有人都不打折，为什么这么表示：观察”baseline_T * (1 - discount_level2) * (1 - policy)“

## follow our policy, but give -10% discount for the group doesn't recommend to give discount
policy_dic["Our Policy + Give Negative Discount for No-Discount Group"] = np.mean(revenue_fn(train_data, 0.1, -0.1, 1, policy))

## give everyone -10% discount
policy_dic["Give Everyone Negative Discount"] = np.mean(revenue_fn(train_data, -0.1, 0, 1, np.ones(len(X))))


# In[160]:


# get policy summary table
res = pd.DataFrame.from_dict(policy_dic, orient="index", columns=["利润"])
res["排名"] = res["利润"].rank(ascending=False)
res


# In[ ]:


# 可以看出：非参数模型 est_NonPara 建议的方案获取的利润最高。

