
# coding: utf-8

# # Utilities for extracting output information

# In[1]:


import os
import sys
sys.path.insert(0,'../py')
sys.path.insert(0, '..')
import parameters
import numpy as np
import pandas as pd
import datetime


# In[2]:


def datetime_to_unit(datatime):
    '''Extract datetime information of the data file into 
    hours, minutes, seconds, and milliseconds
    params: 
        datetime 2013-01-27UT05:36:17.48387602
    return:
        [h,m,s,u]
    '''
    times = datatime.split('UT')
    h, m, ss = times[-1].split(':')
    s, u = ss.split('.')
    return [h,m,s,u]

def datetime_to_ms(datetime):
    '''Convert datetime to milliseconds
    params: 
        datetime 2013-01-27UT05:36:17.48387602
    return:
        datetime in ms'''
    datetime = datetime_to_unit(datetime)
    datetime[0] = float(datetime[0])*60*60
    datetime[1] = float(datetime[1])*60
    datetime[2] = float(datetime[2])
    datetime[3] = float(datetime[3])/10**(len(datetime[3]))
    return sum(datetime)

def datetime_diff(datetime2, datetime1):
    '''Difference between the event time and the start of the data collection'''
    return datetime_to_ms(datetime2)-datetime_to_ms(datetime1)


# In[1]:


def extract_output_dataset(data_root, awd_event, site, verbose=False):
    """Extract the output information for each file
    inputs
        data_root   location of the data
        site        site where data was collected
    outputs
        dataset     dictionary mapping each file with the whistler location
    """
    output_path = os.path.join(data_root,site)
    output_file = None
    for file in os.listdir(output_path):
        if file.endswith('.out'):
            output_file = file
            break
    try:
        os.path.exists(output_file)
        with open(os.path.join(output_path, output_file), 'r') as f:
            dataset = {}
            num_line = 0
            lines = f.readlines()
            file_list = []
            last_percent = None
            if verbose:
                print('\nGenerating outputs for %s/%s' %('awdEvent'+str(awd_event),site))
            for line in lines:
                event = {}
                line = line.split('\n') # Remove the '\n' character from each line
                line = line[0].split(' ') 
                line = list(filter(None, line)) # discard empty element in array
                for index in range(2,len(line),2): # store event and probabilities in a dictionary
                    event[line[index]]=line[index+1]
                # save the dictionary
                if line[1] not in file_list: # if file name not in the list
                    dataset[line[1]]=event
                    file_list.append(line[1])
                else:
                    data = dataset[line[1]]
                    event.update(data)
                    dataset[line[1]]=event
                # print progression
                percent = int(num_line*100/len(lines))
                if last_percent != percent:
                    if percent%5==0 and verbose==True:
                        sys.stdout.write("%s%%" % percent)
                        sys.stdout.flush()
                    elif verbose==True:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                    last_percent = percent
                num_line+=1
    except Exception as e:
        print('Error:', e)
    return dataset


def extract_output(awd_event, site, file, verbose=False):
    '''Extract the output information of a file and return the file and the file time format
        as well as the time diffence between the detection of a whistler as per AWDA and the
        begining of the data collection
        params:
            awd_event
            site
            file
        returns:
            file time format
            time difference 
    '''
    output_location = os.path.join(parameters.database_location, 'awdEvents'+str(awd_event))
    output_data = extract_output_dataset(output_location, awd_event, site,verbose=verbose)[file]
    file_time = file[:27]
    outputs = []
    for key,value in output_data.items():
        outputs.append([round(datetime_diff(key,file_time),5), value])
        outputs = sorted(outputs, key=lambda x:x[0])
    return file_time, output_data, outputs

def probable_output(indices, threshold):
    '''Return the indexes when the measure of detection is above the threshold'''
    return [index for index in indices if int(index[-1]) >= threshold]

