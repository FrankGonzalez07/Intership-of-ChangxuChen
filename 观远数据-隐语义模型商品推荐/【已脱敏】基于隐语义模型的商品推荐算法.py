#!/usr/bin/env python
# coding: utf-8

# # 调包&读数

# In[39]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from numpy.linalg import norm
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# In[4]:


os.chdir('D:\【2019.09.29】workstataion\【2020.11.22】PigHaveDream\【2021.04.21】简历\【2021.06.02】观远数据-推荐系统\基于隐语义模型的书本推荐算法')


# In[5]:


books = pd.read_csv('Books.csv', sep=',', error_bad_lines=False, usecols=[2,3,4,5,6],encoding="latin-1")
print("数据形状:", books.shape)
print("数据列数: ", books.columns)
print("商品种类:",(books.bookISBN.unique().size))

books


# In[6]:


users = pd.read_csv('Users.csv', sep=',', error_bad_lines=False, usecols=[2,3,4], encoding="latin-1")
print("数据形状:", users.shape)
print("数据列数: ", users.columns)
print("用户人数:",(users.user.unique().size))

users


# In[7]:


ratings = pd.read_csv('UserEvents.csv', sep=',', error_bad_lines=False, usecols=[1,2,3], encoding="latin-1")

print("数据形状:", users.shape)
print("数据列数: ", users.columns)
print("用户人数: ",ratings.user.unique().size)
print("商品种类: ",ratings.bookId.unique().size)

ratings


# In[8]:


print("最受欢迎商品:")
usersperbook = ratings.bookId.value_counts()
usersperbook.head(10)


# In[9]:


print("最活跃用户:")
booksperuser = ratings.user.value_counts()
booksperuser.head(10)


# In[12]:


ratings.impression.value_counts(sort=False).plot(kind='bar')
plt.title('用户评价统计')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# # 数据清洗

# In[13]:


# 用户评价转码
ratings["impression"]=ratings["impression"].map({"dislike":1,"view":2,"interact":3,"like":4,"add to cart":5,"checkout":6})


# In[14]:


average_rating = pd.DataFrame(ratings.groupby('bookId')['impression'].mean())
average_rating['ratingCount'] = pd.DataFrame(ratings.groupby('bookId')['impression'].count())
average_rating.sort_values('ratingCount', ascending=False).head(20)


# In[63]:


ratings


# # 以用户ID为列的数据透视表

# In[61]:


# 构建关于用户的数据透视表
ratings_pivot = ratings.pivot(index='user', columns='bookId', values="impression").fillna(0)
print(ratings_pivot.shape)
ratings_pivot


# In[16]:


# 将数据透视表的行名与列名分别生成 list
user_list=list(ratings_pivot.index)
book_list=list(ratings_pivot.columns)


# In[67]:


print("用户ID(Unique):",len(user_list))
user_list


# In[68]:


print("商品ID(Unique):", len(book_list))
book_list


# In[73]:


ratings_pivot


# In[72]:


ratings_pivot.values


# # 对数据透视表做稀疏化处理

# In[69]:


R = coo_matrix(ratings_pivot.values) # 将矩阵做稀疏化处理，可以显示每个用户对自己所购买的具体每个商品的具体评价分数，
# 但是会将每个用户重新编码，具体详见 print(R)

print(ratings_pivot.shape)
print ("R Shape:", R.shape)
print ("R Columns:", R.col) # 
print ("R Rows:",R.row)     # 


# In[92]:


print(R)
# 表示每个用户对自己所购买的具体每个商品的具体评价分数


# In[89]:


RColList = R.col.tolist()
RColList.sort()
print(len(RColList)) # 按照每个商品编码的购买记录条数，共141081条
RColList


# In[90]:


RRowList = R.row.tolist()
RRowList.sort()
print(len(RRowList)) # 按照每个用户编码的购买记录条数，共141081条
RRowList


# # 计算每个商品（行）关于用户（列）的RMSE矩阵

# In[94]:


M,N = R.shape
# M：用户数量（行）；N：商品数量（列）
print("用户数量（行）：",M)
print("商品数量（列）：",N)


# In[30]:


K = 3


# In[32]:


P = np.random.rand(M,K)  # 生成 M 行 K 列的0~1的随机数矩阵
Q = np.random.rand(K,N)  # 生成 K 行 N 列的0~1的随机数矩阵


# In[37]:


print(P.shape)
P
# 生成一个 13030*3 的 0~1用户随机数矩阵


# In[38]:


print(Q.shape)
Q
# 生成一个 3*11234 的 0~1商品随机数矩阵


# In[127]:


def error(R,P,Q,lamda=0.02):
    # R：每个用户对自己所购买的具体每个商品的具体评价分数
    # P：一个 13030*3 的 0~1用户随机数矩阵
    # Q：一个 3*11234 的 0~1商品随机数矩阵
    
    ratings = R.data # 提取每个用户对自己所购买的具体每个商品的具体评价分数，一个长度为141081的栈
    rows = R.row     # 按照每个用户编码的购买记录条数
    cols = R.col     # 按照每个商品编码的购买记录条数
    e = 0
    for ui in range(len(ratings)): # len(ratings) == 141081（即按照每个用户编码的购买记录条数）
        rui = ratings[ui] # 141081条的商品购买记录中第 ui 个用户的评价分数
        u = rows[ui]  # 141081条的商品购买记录中第 ui 个用户的编码
        i = cols[ui]  # 141081条的商品购买记录中第 ui 个用户所购买商品的编码
        if rui > 0:
            e = e + pow(rui-np.dot(P[u,:], Q[:,i]),2) + lamda*(pow(norm(P[u,:]),2) + pow(norm(Q[:,i]),2))  
            # norm(P[3,:])：表示每个数的平方和，再开方
            # P[u,:]表示第 u 个用户的三列随机数；
            # Q[:,i]表示第 i 个商品的三列随机数
            # pow(a,b) = a^b
    return e


# In[148]:


# np.dot 表示点阵乘积
x = np.array([1,2,3])
y = np.array([1,2,3])
result = np.dot(x, y)
result


# In[149]:


rmse = np.sqrt(error(R,P,Q)/len(R.data))
rmse


# In[150]:


def SGD(R, K, lamda=0.02,steps=10, gamma=0.001):
    
    M,N = R.shape
    P = np.random.rand(M,K)
    Q = np.random.rand(K,N)
    
    rmse = np.sqrt(error(R,P,Q,lamda)/len(R.data))
    print("Initial RMSE: "+str(rmse))
    
    for step in range(steps):
        for ui in range(len(R.data)):
            rui=R.data[ui]
            u = R.row[ui]
            i = R.col[ui]
            if rui>0:
                eui=rui-np.dot(P[u,:],Q[:,i])
                P[u,:]=P[u,:]+gamma*2*(eui*Q[:,i]-lamda*P[u,:])
                Q[:,i]=Q[:,i]+gamma*2*(eui*P[u,:]-lamda*Q[:,i])
        rmse = np.sqrt(error(R,P,Q,lamda)/len(R.data))
        if rmse<0.5:
            break
    print("Final RMSE: "+str(rmse))
    return P,Q


# In[151]:


P,Q=SGD(R,K=3,gamma=0.0007,lamda=0.01, steps=100)


# In[152]:


all_user_ratings =np.matmul(P, Q) # 返回两个矩阵的乘积
all_user_ratings


# In[153]:


all_user_ratings_df = pd.DataFrame(np.round(all_user_ratings,4),columns=book_list, index=user_list)
all_user_ratings_df.shape


# In[154]:


all_user_ratings_df.head(10)


# In[ ]:


all_user_ratings_df.to_csv('output.csv', sep=',', encoding='utf-8')


# In[155]:


all_user_ratings_df1=all_user_ratings_df.transpose()
all_user_ratings_df1.head()


# In[156]:


top_five_df= all_user_ratings_df1[99].sort_values(ascending=False)


# In[157]:


top_five_df.head(5)

