#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv("w2v_final_data.csv")
df = df.drop(columns=['Caption'])
#data shuffle
df = df.sample(frac=1).reset_index(drop=True)


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Label',axis=1), 
                                                    df['Label'], test_size=0.20, 
                                                    random_state=101)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)


# In[4]:


print(classification_report(y_test,predictions))


# In[ ]:




