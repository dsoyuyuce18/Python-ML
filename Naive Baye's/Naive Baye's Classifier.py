#!/usr/bin/env python
# coding: utf-8

# ###### ------------------------------------------------
# ## Engr 421 - HW3
# ## Naive Bayes Classifier
# ## Doğukan Soyuyüce
# ## 69331
# ##### --------------------------------------------------
# 

# 

# In[299]:


### Import necessary libraries
import math
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns


# In[250]:


#Read given real life datasets.
x = pd.read_csv('hw02_data_set_images (1).csv',header=None)
y = pd.read_csv('hw02_data_set_labels (1).csv',header=None)
x.shape


# In[225]:


def safelog(x):
    return(np.log(x + 1e-100))


# In[226]:


#split dataaset into train and test sets.
train_a, test_a = x[:25], x[25:40]
train_b, test_b = x[40:65], x[65:79]
train_c, test_c = x[79:104], x[104:118]
train_d, test_d = x[118:143], x[143:157]
train_e, test_e = x[157:182], x[182:196]


# In[251]:


frames1 = [train_a,train_b,train_c,train_d,train_e]
frames2 = [test_a,test_b,test_c,test_d,test_e]

x_train = pd.concat(frames1,ignore_index=True)
x_test=pd.concat(frames2,ignore_index=True)

train_set=x_train.to_numpy()
test_set=x_test.to_numpy()


# In[292]:


y_train_a, y_test_a = y[:25], y[25:40]
y_train_b, y_test_b = y[40:65], y[65:79]
y_train_c, y_test_c = y[79:104], y[104:118]
y_train_d, y_test_d = y[118:143], y[143:157]
y_train_e, y_test_e = y[157:182], y[182:196]



frames1 = [y_train_a,y_train_b,y_train_c,y_train_d,y_train_e]
frames2 = [y_test_a,y_test_b,y_test_c,y_test_d,y_test_e]

y_train = pd.concat(frames1,ignore_index=True)
y_test=pd.concat(frames2,ignore_index=True)
Y_train=y_train.to_numpy()
Y_test=y_test.to_numpy()


# In[293]:


y_train[y_train=='A']=1
y_train[y_train=='B']=2
y_train[y_train=='C']=3
y_train[y_train=='D']=4
y_train[y_train=='E']=5

y_truth=y_train[0].to_numpy()

y_truth.shape


# In[294]:


K = np.max(y_truth)
N = 125;


# In[295]:


# Find means in order to calculate pcd.
a_sum = train_a.sum(axis=0) / 25
b_sum = train_b.sum(axis=0) / 25
c_sum = train_c.sum(axis=0) / 25
d_sum = train_d.sum(axis=0) / 25
e_sum = train_e.sum(axis=0) / 25


# In[296]:


pcd = pd.concat([a_sum.to_frame(),b_sum.to_frame(),c_sum.to_frame(),d_sum.to_frame(),e_sum.to_frame()],axis=1)

priors = [np.mean(y == (c+1)) for c in range(K)]


# In[304]:


#Calculate score functions for train set.
score_1 = np.sum(train_a*safelog(a_sum) +((1-train_a)*safelog(1-a_sum)))+ safelog(1/5)
score_2 = np.sum(train_b*safelog(b_sum) + ((1-train_b)*safelog(1-b_sum)))+ safelog(1/5)
score_3 = np.sum(train_c*safelog(c_sum) + ((1-train_c)*safelog(1-c_sum)))+ safelog(1/5)
score_4 = np.sum(train_d*safelog(d_sum) + ((1-train_d)*safelog(1-d_sum)))+ safelog(1/5)
score_5 = np.sum(train_e*safelog(e_sum) + ((1-train_e)*safelog(1-e_sum)))+ safelog(1/5)

score = pd.concat([score_1,score_2,score_3,score_4,score_5],axis=1)

score.shape
score_1.shape


# In[305]:


# Confusion matrix for train set.
predict = np.argmax(score.to_numpy(), axis = 1) +1
confusion_matrix = pd.crosstab(predict[0:125], y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[306]:


# Confusion matrix for test set.
t_score_1 = np.sum(test_a*safelog(a_sum) +((1-test_a)*safelog(1-a_sum)))+ safelog(1/5)
t_score_2 = np.sum(test_b*safelog(b_sum) + ((1-test_b)*safelog(1-b_sum)))+ safelog(1/5)
t_score_3 = np.sum(test_c*safelog(c_sum) + ((1-test_c)*safelog(1-c_sum)))+ safelog(1/5)
t_score_4 = np.sum(test_d*safelog(d_sum) + ((1-test_d)*safelog(1-d_sum)))+ safelog(1/5)
t_score_5 = np.sum(test_e*safelog(e_sum) + ((1-test_e)*safelog(1-e_sum)))+ safelog(1/5)

score2 = pd.concat([t_score_1,t_score_2,t_score_3,t_score_4,t_score_5],axis=1)


# In[307]:


predict = np.argmax(score2.to_numpy(), axis = 1) +1
confusion_matrix = pd.crosstab(predict[0:125], y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[ ]:




