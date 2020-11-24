#!/usr/bin/env python
# coding: utf-8

# # Engr421 Homework1 -   Dogukan Soyuyüce
# 

# # Importing necessary libraries.

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 
def safelog(x):
    return(np.log(x + 1e-100))


# # Defining means and covariances in order to produce random data points

# In[3]:


np.random.seed(69331)
# I will produce my random data through using np.random.multivariate_normal function.
# I produced class_means and class_covariances in order to use in that function. I will have random data points 
# in that given mean and covariances intervals

class_means = np.array([[+0.0, +2.5], [-2.5, -2.0], [+2.5, -2.0]])
class_covariances = np.array([[[+3.2, +0.0], [+0.0, +1.2]],
                              [[+1.2, -0.8], [-0.8, +1.2]],
                              [[+1.2, +0.8], [+0.8, +1.2]]])

# I assigned class sized 120 for class1 , 90 for class2 and 90 for class3 .
class_sizes = np.array([120, 90, 90])


# In[4]:


# I generated my random data sets using class means and covariances with help pf random.multivariate_normal function.
points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_covariances[2,:,:], class_sizes[2])
X = np.vstack((points1, points2, points3))

# We have corresponding labels for each data we have. I created my label set. 1 for class1 , 2 for class2, 3 for class3.
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))


# In[5]:


# I stacked my data and labels together and I had a dataset ready to use for training.
df = np.hstack((X, y[:, None]))


#  # Visualizing data points using plt.plot

# In[6]:


# I visualized my data in 3 different colors.
plt.figure(figsize = (10, 10))
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 15)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 15)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 15)

# I labeled my x and y coordinates.
plt.xlabel("x1 values")
plt.ylabel("x2 values")


# # Define x_train and y_train  matrices.

# In[7]:


# I defined my x_train and y_train values in order to implement softmax algorithm on it.
X = df[:,[0, 1]]
y_truth = df[:,2].astype(int)

# I defined the total number of samples and number of classes.
K = np.max(y_truth)
N = df.shape[0]

# K containing a single 1 for the correct class and 0 elsewhere
Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth - 1] = 1
Y_truth


# # Parameter estimation .

# In[8]:


# I created my sample_means 
sample_means= [np.mean(X[y==(c+1)],axis=0) for c in range(K)]
sample_means


# In[9]:


# I created my sample_covariances
sample_covariances =[np.cov(X[y==(c+1)]) for c in range(K)]


# In[10]:


class_priors = [np.mean(y == (c + 1)) for c in range(K)]
class_priors


# # Defining algortihms (gradient descent to determine weights)

# In[11]:


# define the softmax function
def softmax(X, W, w0):
    scores = np.matmul(np.hstack((X, np.ones((N, 1)))), np.vstack((W, w0)))
    scores = np.exp(scores - np.repeat(np.amax(scores, axis = 1, keepdims = True), K, axis = 1))
    scores = scores / np.repeat(np.sum(scores, axis = 1, keepdims = True), K, axis = 1)
    return(scores)


# In[12]:


# define the gradient functions
def gradient_W(X, y_truth, y_predicted):
    return(np.asarray([-np.sum(np.repeat((Y_truth[:,c] - Y_predicted[:,c])[:, None], X.shape[1], axis = 1) * X, axis = 0) for c in range(K)]).transpose())

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum(Y_truth - Y_predicted, axis = 0))


# In[13]:


# set parameters
eta = 0.01
epsilon = 1e-3


# In[14]:


# I defined my weights (namely w and w0) randomly between the range of -0.01 and 0.01 . 
np.random.seed(69331)
W = np.random.uniform(low = -0.01, high = 0.01, size = (X.shape[1], K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))


# In[15]:


#Using gradient descent I learnt the real values of my weights.
iteration = 1
objective_values = []
while 1:
    Y_predicted = softmax(X, W, w0)

    objective_values = np.append(objective_values, -np.sum(Y_truth * safelog(Y_predicted)))

    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(X, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)

    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break

    iteration = iteration + 1
print(W)
print(w0)


# # Plot iterarion graph

# In[16]:


# Objective function is plotted and observed that error is decreasing while iteration increases.
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# # Calculating confusion matrix

# In[17]:


# calculate confusion matrix which demonstrates the accuracy of our algorithm.
y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# # Visualize final result. using contourf function

# In[22]:


# Visualizing the final result.
x1_interval = np.linspace(-8, +8, 1201)
x2_interval = np.linspace(-8, +8, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))
for c in range(K):
    discriminant_values[:,:,c] = W[0, c] * x1_grid + W[1, c] * x2_grid + w0[0, c]

A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:,:,0] = A
discriminant_values[:,:,1] = B
discriminant_values[:,:,2] = C

plt.figure(figsize = (10, 10))
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "r.", markersize = 10)
plt.plot(X[y_truth == 2, 0], X[y_truth == 2, 1], "g.", markersize = 10)
plt.plot(X[y_truth == 3, 0], X[y_truth == 3, 1], "b.", markersize = 10)
plt.plot(X[y_predicted != y_truth, 0], X[y_predicted != y_truth, 1], "ko", markersize = 20, fillstyle = "none")

plt.contour(x1_grid, x2_grid, A-B, levels = 0)
plt.contour(x1_grid, x2_grid, A - C, levels = 0, )
plt.contour(x1_grid, x2_grid, B - C, levels = 0, )


plt.xlabel("x1 values")
plt.ylabel("x2 values")


# In[ ]:





# In[ ]:




