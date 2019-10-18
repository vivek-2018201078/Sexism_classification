#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
import pandas as pd

import math

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


# In[2]:


## Read csv into pandas DataFrame and drop memtioned columns

def Read_DataFile(filepath, Columns_to_drop=[]):
    df = pd.read_csv(filepath)
    df = df.drop(columns=Columns_to_drop)
    return df


# In[3]:


## Merge multiple datafiles into single file (all files must have same columns)

def Merge_DataFiles(filepath_list, Output_filepath):
    
    merged_df = Read_DataFile(filepath_list[0])
    for i in range(1,len(filepath_list)):
        filepath = filepath_list[i]
        temp_df = Read_DataFile(filepath)
        merged_df = pd.concat([merged_df, temp_df])
    
    merged_df.to_csv(Output_filepath)
    return merged_df


# In[4]:


## Data Preprocessing on caption, comments, hashtags

def Data_Processing(data):
    
    Column_list = ["Hashtags", "Tags_Len", "Caption_Tokens", "Cap_Tokens_Len", "Comments_Tokens", "Com_Tokens_Len"]
    processed_df = pd.DataFrame(columns=Column_list)
    Req_Columns = ["text","comments","hashtags"]
    Req_Data = data[Req_Columns]
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9_#]+')
    stop_words = set(stopwords.words('english'))
    
    for row in range(Req_Data.shape[0]):
        
        Hashtags = set()
        Caption = []
        Comments = []
        
        for col in Req_Columns:
            
            text_content = Req_Data.iloc[row][col]
            
            if type(text_content) != str:
                continue

            elif col=="hashtags":

                ## Tokenize with delimiter space, No further processing for hashtags
                tokens = text_content.split()
                
                for token in tokens:
                    if token.startswith('#'):
                        Hashtags.add(token)
            else:

                ## Tokenize with delimiter space
                tokens = tokenizer.tokenize(text_content)

                for token in tokens:

                    ## Remove numbers
                    if token.isnumeric():
                        continue
                    
                    ## Remove hashtags from caption, comments and add them to hashtags list
                    elif token.startswith('#'):
                        Hashtags.add(token)
                    
                    ## Remove stopwords
                    elif token.casefold() in stop_words:
                        continue
                    else:
                        if col == "text":
                            Caption.append(token.casefold())
                        else:
                            Comments.append(token.casefold()) 
        
        Hashtag_string = ' '.join(str(tag) for tag in Hashtags)
        Caption_string = ' '.join(str(c) for c in Caption)
        Comments_string = ' '.join(str(c) for c in Comments)
        
        temp_df = pd.DataFrame([[Hashtag_string,len(Hashtags),Caption_string,len(Caption),Comments_string,len(Comments)]], columns=Column_list)
       
        if row==0:
            processed_df = temp_df
        else:
            processed_df = pd.concat([processed_df, temp_df], ignore_index=True)
        
    return processed_df
        


# In[5]:


def Filter_Data(Data, Hashtags_Threshold=1, Caption_Threshold=10, Comments_Threshold=10):
    
    print("Original data shape: ", Data.shape)
    index_to_drop = []
    
    for row in range(Data.shape[0]):
        
        row_entry = Data.iloc[row] 
        flag = 0
        
        if row_entry['Tags_Len'].item() < Hashtags_Threshold:
            flag = 1
        elif row_entry['Cap_Tokens_Len'].item() < Caption_Threshold:
            flag = 1
        elif row_entry['Com_Tokens_Len'].item() < Comments_Threshold:
            flag = 1
        else:
            continue
            
        if flag==1:
            index_to_drop.append(Data.index[row])
            
    Filtered_Data = Data.drop(index=index_to_drop)      
    print("Filtered data shape: ", Filtered_Data.shape)
    
    return Filtered_Data
    


# In[6]:


Files_to_merge = ["./Data/everydaysexism.csv","./Data/genderbias.csv","./Data/genderstereotype.csv","./Data/heforshe.csv",
                  "./Data/mencallmethings.csv","./Data/metoo.csv","./Data/misogynist.csv","./Data/notallmen.csv",
                  "./Data/questionsformen.csv","./Data/slutgate.csv","./Data/wagegap.csv","./Data/weareequal.csv",
                  "./Data/womenareinferior.csv","./Data/workplaceharassment.csv","./Data/yesallwomen.csv"]

merged_df = Merge_DataFiles(Files_to_merge, "./Data/Merged_Data.csv")
print("Merged Data Shape: ",merged_df.shape)

Processed_Df = Data_Processing(merged_df)
print("Processed Data Shape: ", Processed_Df.shape)
Processed_Df.to_csv("./Data/Processed_Data.csv")




# In[25]:


Filtered_Data_Analysis = []

for i in range(1,11,1):    
    for j in range(5,11,1):     
        for k in range(5,11,1):
            
            Filtered_Data = Filter_Data(Processed_Df,i,j,k)
            row_entry = np.array([int(i), int(j), int(k), int(Filtered_Data.shape[0])])
            Filtered_Data_Analysis.append(row_entry)
            
            print(row_entry)
            print("======================")

print("========= Filtered_Data_Analysis ============")
print(Filtered_Data_Analysis)
        


# In[26]:


Filtered_Data_Analysis = pd.DataFrame(Filtered_Data_Analysis)
Filtered_Data_Analysis.to_csv("./Data/Filtered_Data_Analysis.csv", header=["Hashtags_Threshold", 
                                                                          "Caption_Len_Threshold",
                                                                          "Comments_Len_Threshold",
                                                                          "Filtered_Data_Size"])


# In[29]:


Filtered_Data = Filter_Data(Processed_Df,10,10,10)
Filtered_Data.to_csv("./Data/Filtered_Data.csv")


# In[ ]:


## Merge all files of different hashtags into one file, Drop unnecessary columns
## Remove non_english text(caption, hashtags, comments) from the post
## Tokenize text of each post
## Remove punctuations(except delimiters used), emojis, numbers, stopwords from text(caption, comments)
## Data Analysis: length_of_text vs no_of_posts (Zipf distribution), no_of_hashtags vs no_of_posts,    
##                Distribution of posts as per hashtags
## Remove posts having text of length less than particular threshold



# In[ ]:




