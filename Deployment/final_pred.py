#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests


# In[2]:


url='http://localhost:9696/predict'


# In[3]:


customer= {
  "customerid": "8879-zkjof",
  "gender": "female",
  "seniorcitizen": 0,
  "partner": "no",
  "dependents": "no",
  "tenure": 41,
  "phoneservice": "yes",
  "multiplelines": "no",
  "internetservice": "dsl",
  "onlinesecurity": "yes",
  "onlinebackup": "no",
  "deviceprotection": "yes",
  "techsupport": "yes",
  "streamingtv": "yes",
  "streamingmovies": "yes",
  "contract": "one_year",
  "paperlessbilling": "yes",
  "paymentmethod": "bank_transfer_(automatic)",
  "monthlycharges": 79.85,
  "totalcharges": 3320.75
}


# In[4]:


customer


# In[12]:


response=requests.post(url,json=customer).json()
response


# In[14]:


if response['churn']==True:
    print('sending discount to customer-ID: %s' %('xyz-123'))
else:
    print('not sending discount to customer-ID: %s' %('xyz-123'))


# In[ ]:




