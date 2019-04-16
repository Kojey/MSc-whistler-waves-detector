
# coding: utf-8

# In[1]:


import sys
import os
import time

sys.path.insert(0,'../')
sys.path.insert(0,'../py')

import parameters
import utilities
import spectrogram_utilities
import output_utilities
import spectrogram_output_visualiser

import numpy as np
import pandas as pd
import h5py


# In[4]:


def whistler_cuts_db(awd_event, site, files, database_name,verbose=False):
    '''Extract the whistler cuts abd store them in a h5py database'''
    start = time.time()
    # create h5py database
    utilities.init_h5py(database_name)
    # load database
    database = h5py.File(utilities.get_h5py_path(database_name), 'r+')

    if verbose:
        print('\nGenerating whistler cuts database for %s/%s' %('awdEvent'+str(awd_event),site))
        last_percent = None
        num_file = 0
    for file in files:
        indices, spectrogram, cuts = spectrogram_output_visualiser.whistler_cut(awd_event, site, file, 10)
        i = 0
        for cut in cuts:
            spec = spectrogram[cut[0]:cut[1],cut[2]:cut[3]] # extract portion of interest in the spectrogram 
            dataset_name = file.split(site)[0]+str(i)
            file_dataset = database.create_dataset(dataset_name,spec.shape,np.float32, compression="gzip", data=spec)
            file_dataset.attrs['pb'] = indices[i]
            i += 1
        if verbose:
            percent = int(num_file*100/len(files))
            if last_percent != percent:
                if percent%10==0:
                    sys.stdout.write("%s%%" % percent)
                    sys.stdout.flush()
                else:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                last_percent = percent
            num_file+=1
    database.close()
    end = time.time()
    if verbose:
        print("\nRuntime: {:.2f} seconds".format(end - start))


# In[8]:


def load_whistler_cuts_db(awd_event, site, database_name, verbose=False):
    '''Load whistler cuts from database
    returns:
        array of whistler
    '''
    start = time.time()
    data = []
    # load database
    database = h5py.File(utilities.get_h5py_path(database_name), 'r+')
    files = list(database.keys())
    if verbose:
        print('\nLoading whistler cuts from database for %s/%s' %('awdEvent'+str(awd_event),site))
        last_percent = None
        num_file = 0
    for file in files:
        file_data = np.empty(database[file].shape)
        database[file].read_direct(file_data)
        file_data = file_data.flatten()
        data.append(file_data)
        if verbose:
            percent = int(num_file*100/len(files))
            if last_percent != percent:
                if percent%10==0:
                    sys.stdout.write("%s%%" % percent)
                    sys.stdout.flush()
                else:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                last_percent = percent
            num_file+=1
    data = np.asarray(data, dtype=np.float32)
    database.close()
    end = time.time()
    if verbose:
        print("\nRuntime: {:.2f} seconds".format(end - start))
    return data

