
# coding: utf-8

# In[1]:


import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from scipy import signal
from scipy.stats import norm, pearsonr
from itertools import combinations
from scipy.ndimage.filters import gaussian_filter



# In[2]:


# database full path
database_name = 'whistlers.h5'
database_location = os.path.join(os.getcwd().split(os.environ.get('USER'))[0],os.environ.get('USER'), 'wdml', 'Data')
database_path = os.path.join(database_location,database_name)

# data variables
awd_events = 2
sites = ['marion', 'sanae']


# In[3]:


def extract_spectrogram_info(spectrogram_data):
    '''Extract the time, frequency axis values as well as the 
            spectrogram data.
    inputs:
        spectrogram_data: the spectrogram data including the 
        time and frequency information.
    outputs: 
        time: time values
        frequency: frequency values
        spectrogram: spectrogram
    '''
    time = spectrogram_data[0,1:]
    frequency = spectrogram_data[1:,0]
    spectrogram = spectrogram_data[1:,1:]
    return time, frequency, spectrogram

def reshape_spectrogram(f, t, s):
    f = np.asarray(f)
    t = np.asarray(t)
    s = np.asarray(s)
    _t = np.concatenate(([0],t))
    _s = np.concatenate((f[np.newaxis].T,s), axis=1)
    sft = np.vstack((_t,_s))
    return sft


# In[4]:


file_durations = []
site_recording_frq = 257
for awd_event in range(1,awd_events):
    for site in range(len(sites)):
        temp = []
        f  = h5py.File(database_path, 'r')
        grp_wh = f[os.path.join('awdEvents'+str(awd_event), sites[site],'split_dataset')]
        files = list(grp_wh.keys())
        for file in files:
            temp.append(grp_wh[file].shape[1])
        f.close()
        file_durations.append(np.asarray(temp))
site_recording_time = []
gaussian_site_noise = []
gaussian_mean = (0,0)
gaussian_cov = [[0,0],[0,1]]

for site in range(len(sites)):
    site_recording_time.append(file_durations[site].max())
    gaussian_site_noise.append(np.random.multivariate_normal(gaussian_mean, gaussian_cov, 
                                                             (site_recording_frq, site_recording_time[site]))[:,:,0])


# In[ ]:


def vstack_uneven(base_arr, added_arr):
    index = added_arr.shape[1]
    arr = np.zeros(shape=base_arr.shape)
    arr[:,:index] = added_arr + base_arr[:,:index]
    arr[:,index:] = base_arr[:,index:]
    return arr


# In[ ]:


df = []
for awd_event in range(1,awd_events):
    for site in range(len(sites)):
        data = []
        indexes = []
        columns = list(range(site_recording_time[site]*site_recording_frq))
        columns.append('event')
        f  = h5py.File(database_path, 'r+')
        grp = f[os.path.join('awdEvents'+str(awd_event), sites[site],'split_dataset')]
#         grp_split = f.require_group(os.path.join('awdEvents'+str(awd_event), sites[site],'split_dataset_gaussian'))
        files = list(grp.keys())
        # file = files[np.random.randint(len(files))] # select a random sample
        # file = '2013-07-29UT14:22:21.36931914.marion.vr2'
        print('\nGenerating split dataset gaussian for %s/%s' %('awdEvent'+str(awd_event),sites[site]))
        last_percent = None
        num_file = 0
        for num_file in range(len(files)):
            file = files[num_file]
            file_data = np.empty(grp[file].shape)
            grp[file].read_direct(file_data)
            # extract spectrogram
            _t,_f,Sxx = extract_spectrogram_info(file_data)
            # add noise
            Sxx = vstack_uneven(gaussian_site_noise[site], Sxx)
            Sxx = Sxx.flatten()
            # add event boolean
            Sxx = Sxx.tolist()
            Sxx.append(grp[file].attrs['event'])
            # update data and indexes
            data.append(Sxx)
            indexes.append(file)
            # print progress
            percent = int(num_file*100/len(files))
            if last_percent != percent:
                if percent%10==0:
                    sys.stdout.write("%s%%" % percent)
                    sys.stdout.flush()
                else:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                last_percent = percent
        data_frame = pd.DataFrame.from_records(data, indexes, columns=columns)
        database_name = 'whistler_events_'+sites[site]+'.h5'
        database_path = os.path.join(database_location,database_name)
        data_frame.to_hdf(database_path, key=sites[site])
        f.close()
        


# In[37]:




