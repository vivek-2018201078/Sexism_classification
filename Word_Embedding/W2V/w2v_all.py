#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
from gensim.models import KeyedVectors


# In[5]:


model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


# In[42]:


df = pd.read_csv("Final_Labeled_Data.csv")
df_caption = df[['Caption_Tokens']]
df_label = df[['0']]
df_caption.columns = ['Caption']
df_label.columns = ['Label']


# In[43]:


def embedding(caption):
    words = caption.split(' ')
    length = 0
    x = [0] * 300
    for word in words:
        if word in model.vocab:
            length += 1
            x += model[word]
    #print(x)
    x /= length
    return x


# In[44]:


temp = df_caption['Caption'].apply(embedding)
df_emb = pd.DataFrame(temp.values.tolist())
df_final = pd.concat([df_caption, df_emb, df_label], axis = 1)


# In[49]:


df_final.to_csv("w2v_final_data.csv", index=False)


# In[ ]:




