#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator


# In[2]:


def Zipf_Distribution(Data, X_Axis_Col):
    
    No_Posts = Data.shape[0]
    
    Unique_Vals = Data[X_Axis_Col].unique() 
    Axes_Vals = {}
    
    for tag_val in Unique_Vals:
        Posts = Data[Data[X_Axis_Col]==tag_val]
        Axes_Vals[tag_val] = Posts.shape[0]
    
    Axes_Vals = sorted(Axes_Vals.items(), key=operator.itemgetter(0))  
    Axes = list(map(list, zip(*Axes_Vals)))
    
    No_Hashtags_X = Axes[0]
    No_Posts_Y = Axes[1]
    
    return No_Hashtags_X, No_Posts_Y
        


# In[3]:


df = pd.read_csv("./Data/Processed_Data.csv")


# In[4]:


## X-AXIS: No. of unique hashtags in post text content
## Y-AXIS: No. of posts in corpus containing 'x' No. of unique hashtags

No_Hashtags_X, No_Posts_Y = Zipf_Distribution(df, "Tags_Len")

plt.bar(No_Hashtags_X, No_Posts_Y)

plt.xlabel('No of unique hashtags in post')
plt.ylabel('No of posts')

plt.xlim(0,50)
plt.show()


# In[5]:


## Zipf distribution graph : 
## X-AXIS: No. of words in post caption
## Y-AXIS: No. of posts in corpus containing 'x' No. of words in caption

No_Hashtags_X, No_Posts_Y = Zipf_Distribution(df, "Cap_Tokens_Len")

plt.bar(No_Hashtags_X, No_Posts_Y)

plt.xlabel('No. of words in post caption')
plt.ylabel('No of posts')
plt.title('Caption length zipf distribution')

plt.xlim(0,200)
plt.show()


# In[6]:


## Zipf distribution graph : 
## X-AXIS: No. of words in post comments
## Y-AXIS: No. of posts in corpus containing 'x' No. of words in 10 comments

No_Hashtags_X, No_Posts_Y = Zipf_Distribution(df, "Com_Tokens_Len")

plt.bar(No_Hashtags_X, No_Posts_Y)

plt.xlabel('No. of words in post comments')
plt.ylabel('No of posts')

plt.xlim(0,100)
plt.show()


# In[ ]:


## NEXT:

## Filtering Data criteria
#     1. Restrictions on no. of hashtags and length of other text (caption + comments) separately
#     2. Restrictions on no. of hashtags, length of caption, length of comments separately
#     3. Restrictions on total content of post (hashtags + caption + comments)

## Length of total text content of post VS No of posts
## Distribution of posts w.r.t. hashtags used

