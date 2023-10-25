#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression


# In[2]:


data = pd.read_csv('Iris.csv')
data.head()


# In[3]:


data.describe()


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


data.info()


# In[7]:


data["Species"].value_counts()


# In[11]:


sns.FacetGrid(data, hue="Species",height=5).map(plt.scatter,"SepalLengthCm","PetalLengthCm").add_legend()


# In[12]:


x = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].values
y = data[["Species"]].values


# In[13]:


Model = LogisticRegression()
Model.fit(x,y)


# In[15]:


#accuracy
Model.score(x,y).round(2)


# In[16]:


#Predication
Actual = y
predicted = Model.predict(x)


# In[17]:


from sklearn import metrics
print(metrics.classification_report(Actual,predicted))


# In[18]:


print(metrics.confusion_matrix(Actual,predicted))


# In[19]:


predicted = Model.predict([[5.1,3.5,1.4,0.2]])
predicted


# In[ ]:




