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


## https://developers.google.com/machine-learning/guides/text-classification/step-2

def Stats(df):
    
    df["post_text_len"] = df["Cap_Tokens_Len"] + df["Com_Tokens_Len"]

    Tags_mean = np.mean(df["Tags_Len"])
    CaptionLen_mean = np.mean(df["Cap_Tokens_Len"])
    CommentsLen_mean = np.mean(df["Com_Tokens_Len"])
    PostTextlen_mean = np.mean(df["post_text_len"])

    Tags_median = np.median(df["Tags_Len"])
    CaptionLen_median = np.median(df["Cap_Tokens_Len"])
    CommentsLen_median = np.median(df["Com_Tokens_Len"])
    PostTextlen_median = np.median(df["post_text_len"])

    print("Mean no. of hashtags: ", Tags_mean)
    print("Mean length of caption: ", CaptionLen_mean)
    print("Mean length of comments: ", CommentsLen_mean)
    print("Mean length of post text: ", PostTextlen_mean)
    print("\n")
    print("Median no. of hashtags: ", Tags_median)
    print("Median length of caption: ", CaptionLen_median)
    print("Median length of comments: ", CommentsLen_median)
    print("Median length of post text: ", PostTextlen_median)
    print("\n")

    ## https://developers.google.com/machine-learning/guides/text-classification/step-2-5

    ## Step 1 : Calculate the number of samples/number of words per sample ratio
    SN_Ratio_Cap = df.shape[0]/CaptionLen_median
    SN_Ratio_Com = df.shape[0]/CommentsLen_median
    SN_Ratio_PostText = df.shape[0]/PostTextlen_median

    print("SN_Ratio_Cap: ", SN_Ratio_Cap)
    print("SN_Ratio_Com: ", SN_Ratio_Com)
    print("SN_Ratio_PostText: ", SN_Ratio_PostText)


# In[4]:


df = pd.read_csv("./Data/Processed_Data.csv")
Stats(df)


# In[5]:


## X-AXIS: No. of unique hashtags in post text content
## Y-AXIS: No. of posts in corpus containing 'x' No. of unique hashtags

No_Hashtags_X, No_Posts_Y = Zipf_Distribution(df, "Tags_Len")

plt.bar(No_Hashtags_X, No_Posts_Y)

plt.xlabel('No of unique hashtags in post')
plt.ylabel('No of posts')

plt.xlim(0,50)
plt.show()


# In[6]:


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


# In[7]:


## Zipf distribution graph : 
## X-AXIS: No. of words in post comments
## Y-AXIS: No. of posts in corpus containing 'x' No. of words in 10 comments

No_Hashtags_X, No_Posts_Y = Zipf_Distribution(df, "Com_Tokens_Len")

plt.bar(No_Hashtags_X, No_Posts_Y)

plt.xlabel('No. of words in post comments')
plt.ylabel('No of posts')

plt.xlim(0,100)
plt.show()


# In[8]:


## NEXT:

## Filtering Data criteria
#     1. Restrictions on no. of hashtags and length of other text (caption + comments) separately
#     2. Restrictions on no. of hashtags, length of caption, length of comments separately
#     3. Restrictions on total content of post (hashtags + caption + comments)

## Length of total text content of post VS No of posts
## Distribution of posts w.r.t. hashtags used


# In[ ]:




