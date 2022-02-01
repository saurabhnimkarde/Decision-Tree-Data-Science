#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[2]:


fraud = pd.read_csv("Fraud_check.csv")


# In[3]:


fraud.head()


# In[4]:


# Checking the number of rows and columns
fraud.shape


# In[5]:


# Checking for null values
fraud.info()


# In[6]:


# Renaming the columns 

fraud.rename({'Undergrad':'UG','Marital.Status':'MS', 'Taxable.Income':'TI', 'City.Population':'CP', 'Work.Experience':'WE'},axis = 1, inplace = True)


# In[7]:


fraud.head()


# In[8]:


# Categorizing the tax column based on the condition

fraud['TI'] = fraud.TI.map(lambda taxable_income : 'Risky' if taxable_income <= 30000 else 'Good')


# In[9]:


fraud.head()


# In[10]:


# Converting the categorical columns to proper datatypes

fraud['UG'] = fraud['UG'].astype("category")
fraud['MS'] = fraud['MS'].astype("category")
fraud['Urban'] = fraud['Urban'].astype("category")
fraud['TI'] = fraud['TI'].astype("category")


# In[11]:


fraud.dtypes


# In[12]:


# Encoding the categorical columns by using label encoder

label_encoder = preprocessing.LabelEncoder()
fraud['UG'] = label_encoder.fit_transform(fraud['UG'])

fraud['MS'] = label_encoder.fit_transform(fraud['MS'])

fraud['Urban'] = label_encoder.fit_transform(fraud['Urban'])

fraud['TI'] = label_encoder.fit_transform(fraud['TI'])


# In[13]:


fraud


# In[14]:


fraud['TI'].unique()


# In[15]:


fraud['TI'].value_counts()


# In[16]:


# Splitting the data into x and y as input and output

X = fraud.iloc[:,[0,1,3,4,5]]
Y = fraud.iloc[:,2]


# In[17]:


X


# In[18]:


Y


# In[19]:


# Splitting the data into training and test dataset

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.3, random_state = 10)


# In[20]:


#Building a model using C5.O Decision tree classifier

model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, class_weight = 'balanced')
model.fit(x_train,y_train)


# In[21]:


tree.plot_tree(model)


# In[22]:


fn = ['Undergrad',	'Marital.Status',	'City.Population',	'Work.Experience',	'Urban']
cn = ['Taxable_income is Risky', 'Taxable_income is Good']
fig,axes = plt.subplots(nrows = 1, ncols =1, figsize =(4,4), dpi = 300)   
tree.plot_tree(model, feature_names = fn, class_names = cn, filled = True);


# In[23]:


preds = model.predict(x_test)
preds


# In[24]:


pd.Series(preds).value_counts()


# In[25]:


crosstable = pd.crosstab(preds,y_test)
crosstable


# In[26]:


np.mean(preds==y_test)


# In[27]:


print(classification_report(preds,y_test))


# In[28]:


model_cart = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, class_weight = 'balanced')
model_cart.fit(x_train,y_train)


# In[29]:


tree.plot_tree(model_cart)


# In[31]:


preds1 = model_cart.predict(x_test)
preds1


# In[32]:


np.mean(preds1==y_test)


# In[33]:


from sklearn.metrics import f1_score
print(f1_score(preds1,y_test))


# In[ ]:




