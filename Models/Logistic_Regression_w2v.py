#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[19]:


df = pd.read_csv("w2v_final_data.csv")
df = df.drop(columns=['Caption'])


# In[10]:


#data shuffle
df = df.sample(frac=1).reset_index(drop=True)


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Label',axis=1), 
                                                    df['Label'], test_size=0.20, 
                                                    random_state=101)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[32]:


print(classification_report(y_test,predictions))


# In[ ]:




