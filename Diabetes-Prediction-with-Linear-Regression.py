#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# ## Lets Load Boston Housing Data

# In[90]:


from sklearn.datasets import load_diabetes


# In[91]:


diabetes=load_diabetes()


# In[92]:


type(diabetes)


# In[93]:


diabetes.keys()


# ## Lets check the description of the dataset

# In[94]:


print(diabetes.DESCR)


# In[95]:


print(diabetes.data)


# In[96]:


print(diabetes.feature_names)


# ## Preparing the Dataset

# In[97]:


dataset=pd.DataFrame(diabetes.data, columns=diabetes.feature_names)


# In[98]:


dataset.head()


# In[99]:


dataset['Output']=diabetes.target


# In[100]:


dataset.head()


# In[101]:


dataset.info()


# ## Summarizing the Stats of the data

# In[102]:


dataset.describe()


# ## Check for the missing values

# In[103]:


dataset.isnull().sum()


# ## Exploratory Data Analysis

# In[104]:


## Check for Correlation
dataset.corr()


# ## Scatter Plot of the data

# In[105]:


import seaborn as sns
sns.pairplot(dataset)


# In[106]:


plt.scatter(dataset['age'], dataset['Output'])
plt.xlabel('Age')
plt.ylabel('Output')


# In[107]:


plt.scatter(dataset['s5'], dataset['Output'])
plt.xlabel('s5')
plt.ylabel('Output')


# ## Regression Plots

# In[108]:


import seaborn as sns
sns.regplot(x='s5', y='Output', data=dataset)


# In[109]:


import seaborn as sns
sns.regplot(x='bmi', y='Output', data=dataset)


# In[110]:


import seaborn as sns
sns.regplot(x='sex', y='Output', data=dataset)


# ## Separating Independent & Dependent Features

# In[111]:


X=dataset.iloc[:, :-1]
y=dataset.iloc[:, -1]
X.head()


# ## Train & Test Data Split

# In[112]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=42)


# In[113]:


X_train


# In[114]:


y_train


# ## Standardize the dataset for Linear Regression

# In[115]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[116]:


X_train=scaler.fit_transform(X_train)


# In[117]:


## For test set you just need to mention transform, no need of fit_transform
X_test=scaler.transform(X_test)


# In[118]:


X_train


# In[119]:


X_test


# ## Model Training

# In[120]:


from sklearn.linear_model import LinearRegression


# In[121]:


regression=LinearRegression()


# In[122]:


regression.fit(X_train, y_train)


# ## Print the Coefficients & Intercept

# In[123]:


print("Coefficients:", regression.coef_)
print("Intercept:", regression.intercept_)


# ## Check for on which parameters the model has been trained

# In[124]:


regression.get_params()


# ## Prediction with the Test Data

# In[140]:


reg_pred=regression.predict(X_test)


# In[141]:


reg_pred


# ## Lets Plot a Scatter Plot for y_test, reg_pred i.e Output in the dataset Vs Predicted Output for same input features

# In[142]:


plt.scatter(y_test, reg_pred)


# ## Residuals means Actual Output - Predicted Output

# In[144]:


residuals=y_test-reg_pred


# In[145]:


residuals


# In[146]:


sns.displot(residuals, kind="kde")


# ## Scatter Plot for Predictions Vs Residuals for Uniform Distribution

# In[147]:


plt.scatter(reg_pred, residuals)


# In[149]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print("Mean Error:", mean_absolute_error(y_test, reg_pred))
print("MSE:", mean_squared_error(y_test, reg_pred))
print("squrt_MSE:", np.sqrt(mean_squared_error(y_test, reg_pred)))



# ## R Square and Adjusted R Square

# In[150]:


from sklearn.metrics import r2_score
score=r2_score(y_test, reg_pred)
print(score)


# In[151]:


#Display Adjusted R Square
1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# ## New Data Prediction

# In[156]:


diabetes.data[0].reshape(1, -1).shape
diabetes.data[0]


# In[158]:


#Transform the new data
scaler.transform(diabetes.data[0].reshape(1, -1))


# In[159]:


regression.predict(scaler.transform(diabetes.data[0].reshape(1, -1)))


# ## Pickling the model file for deployment

# In[161]:


import pickle


# In[164]:


pickle.dump(regression, open('regmodel.pkl', 'wb'))


# In[165]:


# We can also load the pickled file
pickled_model=pickle.load(open('regmodel.pkl', 'rb'))


# In[166]:


# Lets Predict with Pickled Model
pickled_model.predict(scaler.transform(diabetes.data[0].reshape(1, -1)))

