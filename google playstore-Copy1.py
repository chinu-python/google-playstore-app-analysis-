#!/usr/bin/env python
# coding: utf-8

# In[33]:


# write code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[34]:


import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from collections import Counter


# In[3]:


gp = pd.read_csv('googleplaystore.csv')
gp.head(20)


# In[4]:


gp['Price'].isnull().sum()


# In[ ]:





# In[5]:


gp.shape


# In[6]:


gp.isnull().sum()


# In[ ]:





# In[ ]:





# In[9]:


free= gp['Type'].fillna('free',inplace=True)
gp['Content Rating'].fillna('Everyone',inplace=True)
gp['Current Ver'].fillna('1.0',inplace=True)
gp['Android Ver'].fillna('2.2 And up ', inplace=True)
gp.isnull().sum()


# **1. What is the download rate by categories?**

# In[10]:


gp.Installs[0] 


# In[11]:


gp['Installs']= gp['Installs'].str.replace(',','')
gp['Installs']= gp['Installs'].str.replace('+','')
gp['Installs']= gp['Installs'].str.replace('Free','0')


# In[12]:


gp['Installs']= pd.to_numeric(gp['Installs'])

gp['Installs']


# In[13]:


gp.groupby('Category')['Installs'].sum().plot.bar(x='Category',y='Installs',color=['orange', 'red', 'green', 'blue', 'cyan','orange','black'])


# **2. What is the name of the 15 most downloaded applications?**

# In[14]:



A=gp.groupby('App')['Installs'].sum()
A.sort_values(ascending=False).head(15)


# **3. What is the download rate for paid applications?**

# In[15]:


gp[gp['Type']=='Paid'].head(17).groupby('App')['Installs'].sum().plot.bar(x='App',y='Category',color=['orange', 'red', 'green', 'blue', 'cyan','orange','black'])


# In[ ]:





# **4. Sort by category.**

# In[16]:


gp.sort_values(by=['Category'], inplace=True) # dataframe sorting 


# In[17]:


gp.head()


# **5. What is the download rate and user rating by category?**

# In[18]:


bar=gp.groupby('Category')['Rating'].sum()
bar.head(35).plot.bar(x='Category',y='Rating',color=['orange', 'red', 'green', 'blue', 'cyan','orange','black'])


# In[ ]:





# In[ ]:





# In[21]:





# In[ ]:





# In[23]:


# for dummy variable encoding for Categories
x = gp.drop(['Installs'], axis=1)


# In[24]:


x.head()


# In[25]:


y = gp['Installs']

y.head()


# In[ ]:





# In[ ]:





# **6. Apply Machine learning models?**

# In[26]:


# train and test split

from sklearn.model_selection import train_test_split


# In[27]:


x_train,x_test,y_train,y_test = train_test_split(x,y)


# In[28]:


x_train


# In[29]:


x_test


# In[30]:


y_train


# In[31]:


y_test


# In[ ]:





# In[ ]:





# In[ ]:




