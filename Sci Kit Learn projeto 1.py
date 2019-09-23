#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


USAHousing = pd.read_csv('USA_Housing.csv')


# In[3]:


USAHousing.info()


# In[4]:


USAHousing.head()


# In[5]:


sns.pairplot(USAHousing)


# In[7]:


sns.heatmap(USAHousing.corr())


# In[9]:


USAHousing.columns


# In[10]:


X = USAHousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[11]:


Y = USAHousing['Price']


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.3, random_state=101)


# In[14]:


X_train


# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


lm = LinearRegression()


# In[17]:


lm.fit(X_train, Y_train)


# In[19]:


print(lm.intercept_)


# In[21]:


print(lm.coef_)


# In[22]:


coefs = pd.DataFrame(lm.coef_,X.columns, columns=['Coefs'])


# In[23]:


coefs


# In[24]:


predict = lm.predict(X_test)


# In[30]:


plt.figure(figsize=(12,6))
plt.scatter(Y_test, predict)


# In[31]:


sns.distplot(Y_test-predict)


# In[32]:


from sklearn import metrics


# In[33]:


print('MAE', metrics.mean_absolute_error(Y_test, predict))


# In[34]:


print('MSE', metrics.mean_squared_error(Y_test, predict))


# In[36]:


print('RMSE', np.sqrt(metrics.mean_squared_error(Y_test, predict)))


# In[ ]:




