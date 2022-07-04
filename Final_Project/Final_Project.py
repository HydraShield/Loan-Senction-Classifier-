#!/usr/bin/env python
# coding: utf-8

# In[3]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# *** Since i download dataset now i will import it ***

# In[4]:


df = pd.read_csv('loan_dataset.csv')
df.head()


# In[9]:


df['loan_status'].value_counts()


# In[19]:


df.shape


# # Data Visulisation

# In[14]:


import seaborn as sbn

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sbn.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[15]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sbn.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # DATA PREPROCESSING

# **convert Date Time object**

# In[22]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])


# In[26]:


#we will add new feature dayofweek and weekend(is or not)
df['dayofweek'] = df['effective_date'].dt.dayofweek
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>=4)  else 0)


# **We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4**
# 

# In[27]:


df.head()


# **Convert Categorical features to numerical values**

# In[28]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# **Convert Male to 0 and Femal to 1**

# In[29]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# **Create feature from few data**

# In[30]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# **Final feature x and label y**

# In[34]:


y = df['loan_status'].values
y[:5]


# In[36]:


x = preprocessing.StandardScaler().fit(Feature).transform(Feature)
x[:5]
# Normalising


# In[46]:


#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


# ## Classification Model

# # 1. KNN

# In[47]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

limit = 12
mean_acc = np.zeros((limit-1))
std_acc = np.zeros((limit-1))

for i in range (1,limit):
    knn = KNeighborsClassifier(n_neighbors = i).fit(x_train,y_train)
    yhat= knn.predict(x_test)
    mean_acc[i-1] = metrics.accuracy_score(y_test, yhat)    
    std_acc[i-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[48]:


plt.plot(range(1,limit),mean_acc)
plt.fill_between(range(1,limit),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

knn = KNeighborsClassifier(n_neighbors=mean_acc.argmax()+1).fit(x_train, y_train)


# # 2. DECISION TREE

# In[53]:


from sklearn.tree import DecisionTreeClassifier
import matplotlib.image as mpimg
from sklearn import tree

#Modelling
modeltree = DecisionTreeClassifier(criterion="entropy")
modeltree.fit(x_train, y_train)

#Prediction
predTree = modeltree.predict(x_test)
print(predTree[0:5])
print(y_test[0:5])


# # 3 SUPPORT VECTOR MECHANISM

# In[56]:


import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn import svm

#Modelling
clf = svm.SVC(kernel="sigmoid")
clf.fit(x_train, y_train)

yhat = clf.predict(x_test)
print(yhat[0:10])
print(y_test[0:10])


# # $ Logistic Regression

# In[58]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver="liblinear").fit(x_train,y_train)

yhat = LR.predict(x_test)
yhat_prob = LR.predict_proba(x_test)

print(yhat_prob[0:5])
print(yhat[0:5])
print(y_test[0:5])


# 
# 
# # MODEL EVALUATION

# **first we Load test dataset and Preprocess them**

# In[61]:


Test = pd.read_csv('loan_test.csv')
Test.head()


# In[65]:


Test['loan_status'].value_counts()


# In[66]:


Test.shape


# In[68]:


Test['due_date'] = pd.to_datetime(Test['due_date'])
Test['effective_date'] = pd.to_datetime(Test['effective_date'])

Test['dayofweek'] = Test['effective_date'].dt.dayofweek
Test['weekend'] = Test['dayofweek'].apply(lambda x: 1 if (x>=4)  else 0)


# In[71]:


Test['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
Test_feature = Test[['Principal','terms','age','Gender','weekend']]
Test_feature = pd.concat([Test_feature,pd.get_dummies(Test['education'])], axis=1)
Test_feature.drop(['Master or Above'], axis = 1,inplace=True)
Test_feature.head()


# In[76]:


#Normlising
X = preprocessing.StandardScaler().fit(Test_feature).transform(Test_feature)
Y = Test['loan_status'].values


# In[77]:


X[:5]


# In[78]:


Y[:5]


# In[82]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

knn_pred=knn.predict(X)
jc1=jaccard_score(Y, knn_pred, pos_label = "PAIDOFF")
fs1=f1_score(Y, knn_pred, average='weighted')

tree_pred=modeltree.predict(X)
jc2=jaccard_score(Y, tree_pred, pos_label = "PAIDOFF")
fs2=f1_score(Y, tree_pred, average='weighted')

svm_pred=clf.predict(X)
jc3=jaccard_score(Y, svm_pred, pos_label = "PAIDOFF")
fs3=f1_score(Y, svm_pred, average='weighted')

log_pred=LR.predict(X)
proba=LR.predict_proba(X)
jc4=jaccard_score(Y, log_pred, pos_label = "PAIDOFF")
fs4=f1_score(Y, log_pred, average='weighted')  
ll4=log_loss(Y, proba)

list_jc = [jc1, jc2, jc3, jc4]
list_fs = [fs1, fs2, fs3, fs4]
list_ll = ['NA', 'NA', 'NA', ll4]


# In[83]:


df = pd.DataFrame(list_jc, index=['KNN','Decision Tree','SVM','Logistic Regression'])
df.columns = ['Jaccard']
df.insert(loc=1, column='F1-score', value=list_fs)
df.insert(loc=2, column='LogLoss', value=list_ll)
df.columns.name = 'Algorithm'
df


# In[ ]:





# In[ ]:





# In[ ]:




