
# coding: utf-8

# In[1]:


import os
import sys


# In[2]:


# database full path
database_location = os.path.join(os.getcwd().split(os.environ.get('USER'))[0],os.environ.get('USER'), 'wdml', 'data')
hyp5_location = os.path.join(database_location, 'h5py')

# data variables
awd_events = 2
sites = ['marion', 'sanae']

# psectrogram default parameters
nperseg = 256
noverlap=64
nfft=512
detrend=False # ‘linear’, ‘constant’
scaling='spectrum'
mode='psd' # [‘psd’, ‘complex’, ‘magnitude’, ‘angle’, ‘phase’]

# whistler cut constants
time_lower_boundary = 0.3 # second
time_upper_boundary = 0.7
freq_upper_boundary = 10000 # Hz
freq_lower_boundary = 1500

