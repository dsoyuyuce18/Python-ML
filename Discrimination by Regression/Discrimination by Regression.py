#!/usr/bin/env python
# coding: utf-8

# ## Engr421 HW-2
# ## Discrimination by Regression
# ## Doğukan Soyuyüce
# ## 69331
# 

# ## I imported libraries needed to implement my project.

# In[164]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


# In[165]:


#I did set parameters as given in .pdf file.
eta = 0.01
epsilon = 1e-3


# ## Define the sigmoid function

# In[179]:


# I defined the sigmoid function which is needed for my algorithm. 
def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))


# ## I imported the given real-life datasets with header == None

# In[167]:


x= pd.read_csv('hw02_data_set_images.csv',header=None)
y = pd.read_csv('hw02_data_set_labels (1).csv',header=None)

x.shape 
y.shape


# ### We are given a main dataset and we are asked to assign the first 25 images from each class to the training set and the remaining 14 images to the test set
# 
# ### After splitting data set , I concatinated the train and test sets. Finally I turned them into numpy arrays from pandas data_frames in order to use in my algortihm 

# In[168]:


training_a, test_a = x[:25], x[25:40]
training_b, test_b = x[40:65], x[65:79]
training_c, test_c = x[79:104], x[104:118]
training_d, test_d = x[118:143], x[143:157]
training_e, test_e = x[157:182], x[182:196]


frames1 = [training_a,training_b,training_c,training_d,training_e]
frames2 = [test_a,test_b,test_c,test_d,test_e]

x_train = pd.concat(frames1,ignore_index=True)
x_test=pd.concat(frames2,ignore_index=True)

X_train=x_train.to_numpy()
X_test=x_test.to_numpy()


# In[169]:


y_train_a, y_test_a = y[:25], y[25:40]
y_train_b, y_test_b = y[40:65], y[65:79]
y_train_c, y_test_c = y[79:104], y[104:118]
y_train_d, y_test_d = y[118:143], y[143:157]
y_train_e, y_test_e = y[157:182], y[182:196]



frames1 = [y_train_a,y_train_b,y_train_c,y_train_d,y_train_e,]
frames2 = [y_test_a,y_test_b,y_test_c,y_test_d,y_test_e]

y_train = pd.concat(frames1,ignore_index=True)
y_test=pd.concat(frames2,ignore_index=True)
Y_train=y_train.to_numpy()
Y_test=y_test.to_numpy()


# ### I changed my label dataset from character array into integer array ( A->1 , B->2 etc..)

# In[170]:


y_train[y_train=='A']=1
y_train[y_train=='B']=2
y_train[y_train=='C']=3
y_train[y_train=='D']=4
y_train[y_train=='E']=5


# In[181]:


y_truth=y_train[0].to_numpy()
y_truth


# ### I implemented one hot encoding to my y_train set .

# In[182]:


K = y_train.nunique().astype(int)[0]
N = x.shape[0]

p =125
Y_truth = np.zeros((p, K)).astype(int)
Y_truth[range(p), y_truth.astype(int) -1] = 1

Y_truth


# ### I defined my gradient functions. 

# In[173]:


# define the gradient functions
def gradient_w(X, Y_truth, Y_predicted):
    return(np.asarray([-np.sum(np.repeat((Y_truth[:,c] - Y_predicted[:,c])[:, None], X.shape[1], axis = 1) * X, axis = 0) for c in range(K)]).transpose())

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum(Y_truth - Y_predicted, axis = 0))


# ### I assigned arbitrary values to my w and w0 parameters for starting

# In[185]:


# I defined my weights (namely w and w0) randomly between the range of -0.01 and 0.01 . 
np.random.seed(99)
w = np.random.uniform(low = -0.01, high = 0.01, size = (x_train.shape[1], K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))


# ### Iteration and learning part of my algortihm.

# In[175]:


# learn w and w0 using gradient descent
iteration = 1
objective_values = []
while 1:
    y_predicted = sigmoid(X_train, w, w0)

    objective_values = np.append(objective_values, -np.sum(Y_truth * safelog(y_predicted)))


    w_old = w
    w0_old = w0

    w = w - eta * gradient_w(X_train, Y_truth, y_predicted) 
    w0 = w0 - eta * gradient_w0(Y_truth, y_predicted)

    if np.sqrt(np.sum(w0 - w0_old)**2 + np.sum((w - w_old)**2)) < epsilon:
        break

    iteration = iteration + 1
print(w, w0)


# ### I plotted errors through my iterations. 

# In[176]:


plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# ### I calculated confusion matrix for train data , which should satisfy my algorithm (diagonal 25)

# In[177]:


predict = np.argmax(y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(predict, y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# ### I calculated confusion matrix for test data .

# In[178]:


y_predicted_test = sigmoid(X_test, w, w0)
test_predicted = np.argmax(y_predicted_test, axis = 1) + 1
confusion_matrix = pd.crosstab(test_predicted,Y_test.flatten(), rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)

