
# coding: utf-8

# In[1]:


import os
import sys


# In[ ]:


# database full path
database_name = 'whistlers.h5'
database_location = os.path.join(os.getcwd().split(os.environ.get('USER'))[0],os.environ.get('USER'), 'wdml', 'Data')
database_path = os.path.join(database_location,database_name)

# data variables
awd_events = 2
sites = ['marion', 'sanae']

detrend='linear'
nfft=512
noverlap=64
scaling='spectrum'

