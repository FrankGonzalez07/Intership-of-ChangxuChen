#!/usr/bin/env python
# coding: utf-8

# # 基于Pearson指数的商品（itemCF）协同过滤

# ## 调包&读数

# In[1]:


import os
import pandas as pd
from scipy import sparse


# In[5]:


os.chdir('D:\【2019.09.29】workstataion\【2020.11.22】PigHaveDream\【2021.04.21】简历\【2021.06.02】观远数据-推荐系统\基于Collaborative Filtering的商品推荐算法')


# In[11]:


ratings_raw = pd.read_csv('ratings.csv', encoding='GB18030')
products = pd.read_csv('products.csv', encoding='GB18030')


# In[12]:


ratings_raw


# In[28]:


products


# In[29]:


ratings = pd.merge(ratings_raw,products).drop(['品类','评分时间'],axis = 1)
ratings


# ## 数据清洗

# In[32]:


userRatings_before = ratings.pivot_table(index=['用户ID'],columns=['商品名称'],values='商品评分')
userRatings_before # 显示每一个用户对不同商品的评分


# In[33]:


# 将非缺失值数量少于10个的列删除，并将剩余的缺失值全部用0填充
userRatings_after = userRatings_before.dropna(thresh=10, axis=1).fillna(0,axis=1)  # “thresh=10, axis=1”表示将非缺失值数量少于10个的列删除
userRatings_after


# ## 构建相关系数矩阵

# In[38]:


corrMatrix = userRatings_after.corr(method='pearson')
corrMatrix


# In[37]:


def get_similar(product_title,rating):
    similar_ratings = corrMatrix[product_title]*(rating-2.5)  
      # "corrMatrix[product_title]"表示返回矩阵中列名称为"product_title"的一整列，即与product_title相关的所有商品（包括自身）的相关性
      # 在“*(rating-2.5)”中，之所以减去2.5，是因为总评分为5分，减去其值的一半，这样保证计算出的得分不会超过5，具备参照性
    similar_ratings = similar_ratings.sort_values(ascending=False)
    #print(type(similar_ratings))
    return similar_ratings


# In[40]:


corrMatrix['商品1001']


# In[41]:


corrMatrix['商品1001']*(5-2.5)


# ## 生成用户偏好

# In[46]:


user_1 = [("商品1000",5),("商品36",4),("商品85",2),("商品18",1)]
similar_products = pd.DataFrame()
for product_title,rating in user_1:
    similar_products = similar_products.append(get_similar(product_title,rating),ignore_index = True)

similar_products
# 生成数据为4行，每一行代表user_1的4个偏好商品与其余商品（包括自身）的相关性得分


# In[48]:


similar_products.sum().sort_values(ascending=False).head(10)
# 将 similar_products 默认按列求和，再按降序排列，可以看出商品1000是最符合该用户的商品。


# # 基于余弦相似度的商品协同过滤

# ## 数据清洗

# In[103]:


import pandas as pd
from scipy.spatial import distance


# In[61]:


data = pd.read_csv('cosine.csv')
data


# In[62]:


# 将data转换为宽型
dataWide = data.pivot("用户ID", "商品名称", "购买数量")
dataWide


# In[63]:


# 将缺失值全部替换为0
dataWide.fillna(0, inplace=True)
dataWide


# ## 商品相似度排名

# In[64]:


data_itemcf = dataWide.copy()
data_itemcf


# In[65]:


# 将数据集还原为默认的整型索引，即将“用户ID”还原为单独的 column
data_itemcf = data_itemcf.reset_index()


# In[66]:


data_itemcf


# In[67]:


# 删除“用户ID”列
data_itemcf = data_itemcf.drop('用户ID', axis = 1)


# In[68]:


# 构建商品购买量矩阵
data_itemcf_matrix = pd.DataFrame(index = data_itemcf.columns, columns = data_itemcf.columns)
data_itemcf_matrix


# In[69]:


data_itemcf_matrix.head()


# In[105]:


# 构建基于余弦相似度的商品相似度矩阵
for i in range(0, len(data_itemcf_matrix.columns)):
    for j in range(0, len(data_itemcf_matrix.columns)):
        data_itemcf_matrix.iloc[i,j] = 1 - distance.cosine(data_itemcf.iloc[:,i], data_itemcf.iloc[:,j])


# In[109]:


data_itemcf_matrix


# In[108]:


# 构建每个商品的相似商品表
data_neighbours = pd.DataFrame(index=data_itemcf_matrix.columns,columns=range(1,6))
data_neighbours


# In[112]:


for i in range(0,len(data_itemcf_matrix.columns)):
    data_neighbours.iloc[i,:5] = data_itemcf_matrix.iloc[0:,i].sort_values(ascending=False)[:5].index

# iloc[i,:5]：取第 i 行的前5列
# data_itemcf_matrix.iloc[0:,i].sort_values(ascending=False)[:5]：对第 i 列降序排列后，列出相似度排名前5的商品


# In[113]:


data_neighbours


# ## 用户推荐表

# ### 抽样（耗时较短）

# In[196]:


data = pd.read_csv('cosine.csv')
data


# In[197]:


dataWide = data.pivot("用户ID", "商品名称", "购买数量")


# In[198]:


data_usercf = dataWide.reset_index()
data_usercf


# In[199]:


# 构建一个空的用户商品表
data_usercf_nan = pd.DataFrame(index=data_usercf.index,columns=data_usercf.columns)
data_usercf_nan.iloc[:,:1] = data_usercf.iloc[:,:1]
data_usercf_nan


# In[200]:


# 为了避免计算量激增，选取前200个商品
data_usercf_200 = data_usercf.iloc[:200,:]
data_usercf_nan_200 = data_usercf_nan.iloc[:200,:]


# In[201]:


data_usercf_200


# In[202]:


data_usercf_nan_200


# In[203]:


# 仅抽取 dataframe 中的 values
data_neighbours.product


# In[204]:


data_itemcf


# In[205]:


def getScore(a, b):
    return sum(a*b)/sum(b)


# In[206]:


data_itemcf


# In[207]:


for i in range(0,len(data_usercf_nan_200.index)):
    for j in range(1,len(data_usercf_nan_200.columns)):  # 从第2列开始遍历，是因为多了一列“用户ID”
        user = data_usercf_nan_200.index[i]              # 维护 i 对应的用户ID
        product = data_usercf_nan_200.columns[j]         # 维护 j 对应的商品名称
 
        if data_usercf_200.iloc[i][j] == 1:
            data_usercf_nan_200.iloc[i][j] = 0           # 不统计该商品自身的相似性
        else:
            product_top_names = data_neighbours.loc[product][1:5]   
            # 从相似商品表中抽取该商品对应的、相似度最高的4个商品（因为第0列为该商品本身，所以要从第1列开始抽取）
            
            product_top_sims = data_itemcf_matrix.loc[product].sort_values(ascending=False)[1:5]
            # 从商品相似度矩阵入手，对第 i 列降序排列后，列出相似度排名前4的商品的相似度
            
            user_purchases = data_itemcf.loc[user,product_top_names]
            # 从商品购买量矩阵 data_itemcf 中按照用户查找：该用户购买的相似度最高的4个商品的数量
            
            print (i)
            print (j)
 
            data_usercf_nan_200.iloc[i][j] = getScore(user_purchases,product_top_sims)
            # 最后，将对应的商品购买量与对应的相似度相乘、再求和，除以对应商品的初始相似度之和，就是该商品


# In[211]:


# 获取每个用户推荐度排名前几的商品，构建每个用户的用户推荐表
data_recommend = pd.DataFrame(index=data_usercf_nan.index, columns=['Person','1','2','3','4','5','6'])
data_recommend.iloc[0:,0] = data_usercf_nan.iloc[:,0]  # 将用户推荐表的第一列定为“用户ID”


# In[212]:


data_recommend


# In[216]:


data_usercf_nan


# In[213]:


for i in range(0,len(data_usercf_nan.index)):
    data_recommend.iloc[i,1:] = data_usercf_nan.iloc[i,:].sort_values(ascending=False).iloc[1:7,].index.transpose()
    # 将每一行的推荐度按列排序，再去前6个


# In[222]:


# 展示每个用户推荐度排名前三的商品
data_recommend.iloc[:50,:4]


# ### 全样本（耗时约32mins）

# In[161]:


for i in range(0,len(data_usercf_nan.index)):
    for j in range(1,len(data_usercf_nan.columns)):  
        user = data_usercf_nan.index[i]
        product = data_usercf_nan.columns[j]
 
        if data_usercf.iloc[i][j] == 1:
            data_usercf_nan.iloc[i][j] = 0           
        else:
            product_top_names = data_neighbours.loc[product][1:10]  
            product_top_sims = data_itemcf_matrix.loc[product].sort_values(ascending=False)[1:10]
            user_purchases = data_itemcf.loc[user,product_top_names]
            
            print (i)
            print (j)
 
            data_usercf_nan.iloc[i][j] = getScore(user_purchases,product_top_sims)



# Get the top products
data_recommend = pd.DataFrame(index=data_usercf_nan.index, columns=['Person','1','2','3','4','5','6'])
data_recommend.iloc[0:,0] = data_usercf_nan.iloc[:,0]



# Instead of top product scores, we want to see names
for i in range(0,len(data_usercf_nan.index)):
    data_recommend.iloc[i,1:] = data_usercf_nan.iloc[i,:].sort_values(ascending=False).iloc[1:7,].index.transpose()



# Print a sample
data_recommend.iloc[:50,:4]






