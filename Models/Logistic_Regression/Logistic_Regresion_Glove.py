#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[14]:


df_train= pd.read_csv("TrainData_Glove_Embeddings.csv")
df_test = pd.read_csv("TestData_Glove_Embeddings.csv")
df_train = df_train.drop(columns=['Unnamed: 0', 'Caption_Tokens'])
df_test = df_test.drop(columns=['Unnamed: 0', 'Caption_Tokens'])
df = pd.concat([df_train, df_test])
df = df.sample(frac=1).reset_index(drop=True)


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Label',axis=1), 
                                                    df['Label'], test_size=0.20, 
                                                    random_state=101)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[17]:


print(classification_report(y_test,predictions))


# In[ ]:




