
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


# In[3]:


def spectrogram_cuts_db(awd_event, site, files, database_name,verbose=False):
    '''Extract the whistler and noise cuts and store them in a h5py database'''
    start = time.time()
    # create h5py database
    utilities.init_h5py(database_name)
    # load database
    database = h5py.File(utilities.get_h5py_path(database_name), 'r+')

    if verbose:
        print('\nGenerating whistler and noise cuts database for %s/%s' %('awdEvent'+str(awd_event),site))
        last_percent = None
        num_file = 0
    for file in files:
        indices, spectrogram, spec_cuts, noise_cuts, f_cut_length, t_cut_length = spectrogram_output_visualiser.spectrogram_cut(awd_event, site, file, 10)
        i = 0
        for cut in spec_cuts:
            spec = spectrogram[cut[0]:cut[1],cut[2]:cut[3]] # extract portion of interest in the spectrogram 
            dataset_name = file.split(site)[0]+str(i)
            file_dataset = database.create_dataset(dataset_name,spec.shape,np.float32, compression="gzip", data=spec)
            file_dataset.attrs['pb'] = int(indices[i][-1])
            file_dataset.attrs['evt'] = True
            i += 1
        for noise in noise_cuts:
            spec = spectrogram[noise[0]:noise[1], noise[2]:noise[3]]
            dataset_name = file.split(site)[0]+str(i)
            file_dataset = database.create_dataset(dataset_name,spec.shape,np.float32, compression="gzip", data=spec)
            file_dataset.attrs['pb'] = 0
            file_dataset.attrs['evt'] = False
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
    database.attrs['freq_length']=f_cut_length
    database.attrs['time_length']=t_cut_length
    database.close()
    end = time.time()
    if verbose:
        print("\nRuntime: {:.2f} seconds".format(end - start))


# In[4]:


def load_spectrogram_cuts_db(awd_event, site, database_name, verbose=False):
    '''Load spectrogram cuts from database
    returns:
        array of spectrogram
    '''
    start = time.time()
    data = []
    pb = []
    evt = []
    # load database
    try:
        database = h5py.File(utilities.get_h5py_path(database_name), 'r+')
    except Exception as e:
        # if no database, create the database
        files = utilities.all_files(awd_event, site)
        spectrogram_cuts_db(awd_event, site, files, database_name,verbose=verbose)
        database = h5py.File(utilities.get_h5py_path(database_name), 'r+')
    files = list(database.keys())
    if verbose:
        print('\nLoading spectrogram cuts from database for %s/%s' %('awdEvent'+str(awd_event),site))
        last_percent = None
        num_file = 0
    for file in files:
        file_data = np.empty(database[file].shape)
        database[file].read_direct(file_data)
        file_data = file_data.flatten()
        data.append(file_data)
        pb.append(database[file].attrs['pb'])
        evt.append(database[file].attrs['evt'])
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
    data = np.array(data)
    pb = np.array(pb)
    evt = np.array(evt)
    
    f_cut_length = database.attrs['freq_length']
    t_cut_length = database.attrs['time_length']
    
    database.close()
    end = time.time()
    if verbose:
        print("\nRuntime: {:.2f} seconds".format(end - start))
    return data, pb, evt, f_cut_length, t_cut_length

