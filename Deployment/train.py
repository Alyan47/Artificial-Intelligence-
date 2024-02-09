#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import wget
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score as mi
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
#output_file='model_C=%s.bin' % C
#output_file=f'model_C={C}.bin' 
output_file='model.bin'
output_file

df=pd.read_csv('D:\Applied Data Science & AI\data-week-3.csv')
df.head()

df.dtypes

df.columns=df.columns.str.lower().str.replace(' ','_')

categorical_column=list(df.dtypes[df.dtypes=='object'].index)
for c in categorical_column:
    df[c]=df[c].str.lower().str.replace(' ','_')



df.dtypes

df.totalcharges=pd.to_numeric(df.totalcharges,errors='coerce')

df.totalcharges=df.totalcharges.fillna(0)

df.isnull().sum()

df.churn=(df.churn=='yes').astype('int')


numeric=[
  'tenure', 'monthlycharges',
   'totalcharges'
]


# In[22]:


categorical= [ 'gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']



df_full_train,df_test=train_test_split(df,test_size=0.2,random_state=1)


# In[25]:


df_train,df_val=train_test_split(df_full_train,test_size=0.25,random_state=1)


# In[26]:


df_train=df_train.reset_index(drop=True)
df_val=df_val.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)
df_full_train=df_full_train.reset_index(drop=True)


# In[27]:


y_train=df_train.churn.values
y_val=df_val.churn.values
y_test=df_test.churn.values


# In[28]:


del df_train['churn']
del df_val['churn']
del df_test['churn']


# In[29]:


y_full_train=df_full_train.churn.values


# In[30]:


df_full_train.churn.value_counts(normalize=True)


# In[31]:


global_churn_rate=df.churn.mean()
round(global_churn_rate,2)


# In[32]:




# In[33]:


def Mutual_information(series):
    return mi(series,df_full_train.churn)


# In[34]:


m=df_full_train[categorical].apply(Mutual_information)
m.sort_values(ascending=False)


# In[35]:


df_full_train[numeric].corrwith(df_full_train.churn)


# In[36]:





# In[37]:


train_dict = df_train[categorical+numeric].to_dict(orient='records')


# In[38]:


dv=DictVectorizer(sparse=False)


# In[39]:


dv.fit(train_dict)
X_train=dv.transform(train_dict)


# In[40]:


X_train.shape


# In[41]:


features_name=list(dv.get_feature_names_out())


# In[42]:


val_dict = df_val[categorical+numeric].to_dict(orient='records')


# In[43]:


X_val=dv.transform(val_dict)


# In[44]:


del df_full_train['churn']


# In[69]:


dv1=DictVectorizer(sparse=False)


# In[70]:


full_train_dict=df_full_train[categorical+numeric].to_dict(orient='records')


# In[71]:


dv1.fit(full_train_dict)
X_full_train=dv1.transform(full_train_dict)


# In[72]:


dict_full_test=df_test[categorical+numeric].to_dict(orient='records')
dv1.fit(dict_full_test)


# In[73]:


X_test=dv1.transform(dict_full_test)


# In[74]:





# In[75]:


model1=LogisticRegression()
model1.fit(X_train,y_train)


# In[76]:


y_pred=model1.predict_proba(X_val)[:,1]


# In[89]:


churn_decision=(y_pred>=0.5)
churn_decision


# In[90]:
print('Training The Model')
print('Wait')
model=LogisticRegression()
model.fit(X_full_train,y_full_train)

print('Model Trained')
# In[91]:


y_final_pred=model.predict_proba(X_test)[:,1]


# In[92]:


y_final=(y_final_pred>=0.5)


# In[93]:





# In[94]:

print('Calculating AUC score')
roc_auc_score(y_test,y_final_pred)


# In[95]:


model1,dv1


# In[96]:

print('Saving Model')
import pickle

f_out=open(output_file,'wb')
pickle.dump((dv1,model),f_out)
f_out.close()

with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, model), f_out)
print('Model Saved')





