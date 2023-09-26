#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import os


# In[3]:


cwd = os.getcwd() 
print("Current Wrorking Directory:", cwd)


# In[4]:


os.listdir()


# In[117]:


os.chdir('C:/Users/91980/Downloads/Dream11folder')
os.listdir()


# In[118]:


dfbat=pd.DataFrame()
x=pd.read_csv('Batting.csv')
x.head()


# In[119]:



dfbat= pd.concat([dfbat,x], axis=0 )
print(dfbat.shape)


# In[75]:


dfbat.head()


# In[120]:


dfbat.columns


# In[121]:


dfbat.drop_duplicates()
dfbat.shape


# In[51]:


dfbat.tail()


# In[52]:


dfbat.reset_index(drop=True ,inplace=True)
dfbat


# In[55]:


dfbat= dfbat.drop(['Unnamed: 0' ], axis=1)
dfbat


# In[87]:


dfbowl=pd.DataFrame()
x=pd.read_csv('Bowling.csv')
x.head()


# In[88]:


dfbowl=pd.concat([dfbowl,x],axis=0)
print(dfbowl.shape)


# In[90]:


dfbowl.head()


# In[91]:


dfbowl.columns


# In[92]:


dfbowl.reset_index(drop=True, inplace=True)
dfbowl


# In[93]:


dfbowl.tail()


# In[96]:


dfbowl= dfbowl.drop(['Unnamed: 0' ], axis=1)
dfbowl


# In[104]:


os.getcwd()


# In[105]:


os.chdir('../../')


# In[106]:


dfbowl=dfbowl.replace([np.inf, -np.inf], np.nan)


# In[107]:


dfbowl=dfbowl.dropna(subset=["econrate"], how="all")


# In[101]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import OrdinalEncoder
ohe= OrdinalEncoder()


# In[122]:


dfbat.columns


# In[123]:


train_feature=['batsman', 'runs', 'balls', '4s', '6s', 'SR', '50s', '100s', 'ducks', 'team1', 'team2', 'winner', 'venue']
win_pred_feature=['dr11Score']


# In[133]:


X_train, X_test, y_train, y_test = train_test_split(dfbat[train_feature],dfbat[win_pred_feature], test_size = 0.2, random_state = 42)
X_train_ohe = ohe.fit_transform(X_train)
X_test_ohe = ohe.fit_transform(X_test)
y_train_ohe = ohe.fit_transform(y_train)
y_test_ohe = ohe.fit_transform(y_test)


# In[125]:


#knn model
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=3)


# In[132]:


knn.fit(X_train_ohe,y_train )


# In[134]:


pred = knn.predict(X_test_ohe)


# In[135]:


a = accuracy_score(y_test,pred)
print('The accuracy using KNN is:',format(a*100))


# In[136]:


# Linear Regression model
lr=LinearRegression()
lr.fit(X_train_ohe,y_train)


# In[138]:


y_lr_train_pred = lr.predict(X_train_ohe)
y_lr_test_pred = lr.predict(X_test_ohe)


# In[139]:


y_lr_train_pred


# In[140]:


y_lr_test_pred


# In[144]:


from sklearn.metrics import mean_squared_error ,r2_score
lr_train_mse = mean_squared_error(y_train , y_lr_train_pred)
lr_train_r2 =r2_score(y_train,y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test , y_lr_test_pred)
lr_test_r2 =r2_score(y_test,y_lr_test_pred)


# In[175]:


print("LR MSE (Train): " ,lr_train_mse)
print("LR R2 (Train): " ,lr_train_r2)
print("LR MSE (Test): " ,lr_test_mse)
print("LR MSE (Test): " ,lr_test_mse)


# In[146]:


lr_results =pd.DataFrame([ 'Linear regression' , lr_train_mse ,lr_train_r2 ,lr_test_mse ,lr_test_r2]).transpose()
lr_results.columns =['Method','Training MSE','Training R2','Test MSE','Test R2']


# In[147]:


lr_results


# In[164]:


a= r2_score(y_test, y_lr_test_pred)
print(" Accuracy of linear Regression Model:",format(a*100))


# In[159]:



# Define model
RF_model = RandomForestClassifier(max_depth=40, random_state=142)

# Fit model
RF_model.fit(X_train_ohe, y_train)

#Predict Output 
predicted= RF_model.predict(X_test_ohe)
a = accuracy_score(y_test,predicted)
print('The accuracy using RandomForest classifier:',format(a*100))


# In[158]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2 , random_state=100)
rf.fit(X_train_ohe, y_train)


# In[161]:


y_rf_train_pred = rf.predict(X_train_ohe)
y_rf_test_pred = rf.predict(X_test_ohe)


# In[162]:


from sklearn.metrics import mean_squared_error ,r2_score
rf_train_mse = mean_squared_error(y_train , y_rf_train_pred)
rf_train_r2 =r2_score(y_train,y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test , y_rf_test_pred)
rf_test_r2 =r2_score(y_test,y_rf_test_pred)


# In[163]:


rf_results =pd.DataFrame([ 'Random Forest' , rf_train_mse ,rf_train_r2 ,rf_test_mse ,rf_test_r2]).transpose()
rf_results.columns =['Method','Training MSE','Training R2','Test MSE','Test R2']
rf_results


# In[165]:


a= r2_score(y_test, y_rf_test_pred)
print(" Accuracy of Random Forrest Regressor:",format(a*100))


# In[153]:


DT_model = DecisionTreeRegressor()

# Fit model
DT_model.fit(X_train_ohe, y_train_ohe)

#Predict Output 
predicted= DT_model.predict(X_test_ohe)
predicted=predicted > 0.5
a = accuracy_score(y_test_ohe,predicted)
print('The accuracy using DecisionTreeRegressor is:',format(a*100))


# In[166]:


dfbowl.columns


# In[167]:


train_feature=['bowler', 'overs', 'runs', 'maidens', 'wicket', 'econrate', 'team2', 'winner', 'venue', 'team1', '4 wicket', '5 wicket']
win_pred_feature=['dr11Score']


# In[169]:


dfbowl.dropna(how='any')


# In[168]:


X_train, X_test, y_train, y_test = train_test_split(dfbowl[train_feature],dfbowl[win_pred_feature], test_size = 0.2, random_state = 42)
X_train_ohe = ohe.fit_transform(X_train)
X_test_ohe = ohe.fit_transform(X_test)
y_train_ohe = ohe.fit_transform(y_train)
y_test_ohe = ohe.fit_transform(y_test)


# In[170]:


#Linear Regression Model For Bowling
lr=LinearRegression()
lr.fit(X_train_ohe,y_train)


# In[171]:


y_lr_train_pred = lr.predict(X_train_ohe)
y_lr_test_pred = lr.predict(X_test_ohe)


# In[172]:


y_lr_train_pred


# In[173]:


y_lr_test_pred


# In[174]:


from sklearn.metrics import mean_squared_error ,r2_score
lr_train_mse = mean_squared_error(y_train , y_lr_train_pred)
lr_train_r2 =r2_score(y_train,y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test , y_lr_test_pred)
lr_test_r2 =r2_score(y_test,y_lr_test_pred)


# In[176]:


print("LR MSE (Train): " ,lr_train_mse)
print("LR R2 (Train): " ,lr_train_r2)
print("LR MSE (Test): " ,lr_test_mse)
print("LR MSE (Test): " ,lr_test_mse)


# In[177]:


lr_results =pd.DataFrame([ 'Linear regression' , lr_train_mse ,lr_train_r2 ,lr_test_mse ,lr_test_r2]).transpose()
lr_results.columns =['Method','Training MSE','Training R2','Test MSE','Test R2']


# In[178]:


lr_results


# In[179]:


a= r2_score(y_test, y_lr_test_pred)
print(" Accuracy of linear Regression Model:",format(a*100))


# In[180]:


#Random Forret Model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2 , random_state=100)
rf.fit(X_train_ohe, y_train)


# In[183]:


y_rf_train_pred = rf.predict(X_train_ohe)
y_rf_test_pred = rf.predict(X_test_ohe)


# In[182]:


from sklearn.metrics import mean_squared_error ,r2_score
rf_train_mse = mean_squared_error(y_train , y_rf_train_pred)
rf_train_r2 =r2_score(y_train,y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test , y_rf_test_pred)
rf_test_r2 =r2_score(y_test,y_rf_test_pred)


# In[184]:


rf_results =pd.DataFrame([ 'Random Forest' , rf_train_mse ,rf_train_r2 ,rf_test_mse ,rf_test_r2]).transpose()
rf_results.columns =['Method','Training MSE','Training R2','Test MSE','Test R2']
rf_results


# In[185]:


a= r2_score(y_test, y_rf_test_pred)
print(" Accuracy of Random Forrest Regressor:",format(a*100))


# In[ ]:




