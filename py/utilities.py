
# coding: utf-8

# In[1]:


import sys
import os

sys.path.insert(0,'..')
sys.path.insert(0,'../py')
import parameters
import numpy as np
import h5py


# In[16]:


def all_files(awd_event, site):
    '''Select all datafiles
    params:
        awd_event
        site
    returns:
        datafiles
    '''
    data_location = os.path.join(parameters.database_location, 'awdEvents'+str(awd_event), site, site+'_data')
    files = None
    if os.path.exists(data_location):
        files = [ file for file in os.listdir(data_location) if file.endswith('.vr2')] # only select .vr2 file
    return files

def random_file(awd_event, site):
    '''Select a random datafile
    params:
        awd_event
        site
    returns:
        file name
    '''
    files = all_files(awd_event, site)
    return files[np.random.randint(len(files))]

def create_h5py(h5py_name):
    '''Create h5py database'''
    try:
        f = h5py.File(os.path.join(parameters.hyp5_location, h5py_name), 'w')
    except OSError as e:
        return False
    f.close()
    return True

def delete_h5py(h5py_name):
    if os.path.exists(os.path.join(parameters.hyp5_location, h5py_name)):
        os.remove(os.path.join(parameters.hyp5_location, h5py_name))
        return True
    return False

def get_h5py_path(h5py_name):
    return os.path.join(parameters.hyp5_location, h5py_name)

def init_h5py(h5py_name):
    '''Create database if non existing otherwise delete the existing a create a new one'''
    if os.path.exists(get_h5py_path(h5py_name)):
        delete_h5py(h5py_name)
        create_h5py(h5py_name)
    else:
        create_h5py(h5py_name)

