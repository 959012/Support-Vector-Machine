#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


dataset = pd.read_csv(r"C:\Users\BISWA\Desktop\ML Project\Social_Network_Ads.csv")


# In[4]:


dataset.head()


# In[5]:


dataset.shape


# In[6]:


dataset.columns


# In[8]:


dataset.describe()


# In[11]:


x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.25,random_state = 0)


# In[14]:


from sklearn.preprocessing import StandardScaler


# In[16]:


sc = StandardScaler()


# In[17]:


x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[18]:


from sklearn.svm import SVC


# In[20]:


classifier = SVC(kernel='linear',random_state=0)


# In[21]:


classifier.fit(x_train,y_train)


# In[22]:


y_pre = classifier.predict(x_test)


# In[23]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_pre,y_test)


# In[ ]:





# In[ ]:




