#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


data=pd.read_excel("C:/Users/Dell/Desktop/intern/oasis/car prize/carprize.xlsx")
data


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.isnull().sum()


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


data.shape


# In[10]:


unq=data.CarName.unique()
print(unq)


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


sns.set_style("whitegrid")
plt.figure(figsize=(15,10))
sns.distplot(data.price)
plt.show()


# In[13]:


print(data.corr())


# In[14]:


plt.figure(figsize=(30,20))
correlations=data.corr()
sns.heatmap(correlations,cmap="coolwarm",annot=True)
plt.show()


# In[15]:


predict = "price"
data = data[["symboling", "wheelbase", "carlength", "carwidth", "carheight", "curbweight", "enginesize", "boreratio", "stroke", "compressionratio", "horsepower", "peakrpm", "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[16]:


from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=0)


# In[17]:


print(xtrain)


# In[18]:


print(ytrain)


# In[19]:


print(xtest)


# In[20]:


print(ytest)


# In[23]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)

