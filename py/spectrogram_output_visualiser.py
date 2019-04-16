
# coding: utf-8

# In[3]:


import sys
import os

sys.path.insert(0,'..')
sys.path.insert(0,'../py')

import parameters
import utilities
import spectrogram_utilities
import output_utilities

import numpy as np
from scipy import stats
from sklearn import preprocessing


# In[1]:


def spectrogram_and_output(awd_event, site, file, output=True, verbose=False, zscore=False, method=None, output_distribution='normal'):
    '''Display the spectrogram of the selected file. 
        params:
            awd_event:
            site:
            file:
            zscore: zscore pre-processing
            verbose: show progress of extraction of file outputs
            method: method used for pre-processing (either 'normalize' or 'quantile')
            output_distribution: the distribution used for the quantile method
            output: the AWDA output is added to the processed spectrogram
        returns:
            indices: index of output as per AWDA
            times: time axis of spectrogram
            frequencies: frequency axis of spectrogram
            spectrogram: spectrogram
    '''
    _indices = []
    data_location = os.path.join(parameters.database_location, 'awdEvents'+str(awd_event), site, site+'_data')
    frequencies, times, spectrogram = spectrogram_utilities.spectrogram_from_vr2(data_location, file, site)
    if zscore:
        spectrogram = stats.zscore(spectrogram, axis = 0)
        spectrogram = stats.zscore(spectrogram, axis = 1)
    file_time, output_data, outputs = output_utilities.extract_output(awd_event, site, file, verbose=verbose)
    _t = np.round_(times,decimals=4)
    index = 0
    indices = []
    indexed_output = []
    for o in outputs:
        event_time = np.round(o[0],4)
        # find index of that event in the spectrogram
        index = min(range(len(_t)), key=lambda i: abs(_t[i]-event_time))
        # only process if the index found is new
        if index!=0 and index not in indices:
            indices.append(index)
        indexed_output.append([index,str(list(output_data.keys())[list(output_data.values()).index(o[1])]),
                                   o[0], o[1]])
    #   events = sorted(outputs[:,1], reverse=True)[:len(indices)] # map prob to to event correctly
    if output:
        for index_pb in indexed_output:
            spectrogram[:,index_pb[0]] = np.full(spectrogram[:,index_pb[0]].shape, spectrogram.min())
    if method:
        if method=='normalize':
            spectrogram = preprocessing.normalize(spectrogram)
        elif method=='quantile':
            preprocessing.quantile_transform(spectrogram, output_distribution=output_distribution)
    if verbose:
        print("\nFile name:", file)
        print("[Index; Time (output format); Time (s); Weight]")
        for index in indexed_output:
            print(index)
            
    return indexed_output , times, frequencies, spectrogram

def whistler_cut(awd_event, site, file, threshold, output=False, verbose=False,zscore=True, method='normalize'):
    '''Extract whistler cut out of the spectrogram using the output file oubtained form the AWDA method
    returns:
        spectrogram
        spectrogram boundaries
    '''
    indices, times, frequencies, spectrogram = spectrogram_and_output(awd_event, 
                                            site, file, output, verbose, zscore, method='normalize')
    time_max_index = spectrogram.shape[1] # maximum index on the time axis
    freq_max_index = spectrogram.shape[0] # maximum index on the frequency axis
    time_res = times[-1]/time_max_index   # time resolution
    freq_res = frequencies[-1]/freq_max_index # frequency resolution

    # window cut definition
    time_lower_boundary = parameters.time_lower_boundary # second
    time_upper_boundary = parameters.time_upper_boundary
    freq_upper_boundary = parameters.freq_upper_boundary # Hz
    freq_lower_boundary = parameters.freq_lower_boundary
    
    output_upper_f_index = int(freq_upper_boundary/freq_res)
    output_lower_f_index = int(freq_lower_boundary/freq_res)
    
    spectrograms_boundaries = [] 
    _indices = output_utilities.probable_output(indices, threshold)
    for index in _indices:
        # find maximum time boundary
        output_index = index[0]
        output_upper_t_index, output_lower_t_index = output_index, output_index
        if output_index + int(time_upper_boundary/time_res) <= time_max_index:
            output_upper_t_index += int(time_upper_boundary/time_res)
        else:
            extra = abs(time_max_index - (output_index + int(time_upper_boundary/time_res)))
            output_upper_t_index = time_max_index
            output_lower_t_index -= (int(time_lower_boundary/time_res)+extra)
            spectrograms_boundaries.append([output_lower_f_index,output_upper_f_index,output_lower_t_index,output_upper_t_index])
            continue
        if output_index - int(time_lower_boundary/time_res) >= 0:
            output_lower_t_index -= int(time_lower_boundary/time_res)
        else:
            extra = abs(output_index - int(time_lower_boundary/time_res))
            output_upper_t_index += extra
            output_lower_t_index = 0
        spectrograms_boundaries.append([output_lower_f_index,output_upper_f_index,output_lower_t_index,output_upper_t_index])
    spectrograms_boundaries = np.array(spectrograms_boundaries)
    return _indices, spectrogram, spectrograms_boundaries


# In[ ]:


def spectrogram_cut(awd_event, site, file, threshold, output=False, verbose=False,zscore=True, method='normalize'):
    '''Extract whistler cut out of the spectrogram using the output file oubtained form the AWDA method
    returns:
        spectrogram
        whistlers boundaries
        noise boundaries
    '''
    indices, times, frequencies, spectrogram = spectrogram_and_output(awd_event, 
                                            site, file, output, verbose, zscore, method='normalize')
    time_max_index = spectrogram.shape[1] # maximum index on the time axis
    freq_max_index = spectrogram.shape[0] # maximum index on the frequency axis
    time_res = times[-1]/time_max_index   # time resolution
    freq_res = frequencies[-1]/freq_max_index # frequency resolution

    # window cut definition
    time_lower_boundary = parameters.time_lower_boundary # second
    time_upper_boundary = parameters.time_upper_boundary
    freq_upper_boundary = parameters.freq_upper_boundary # Hz
    freq_lower_boundary = parameters.freq_lower_boundary
    
    output_upper_f_index = int(freq_upper_boundary/freq_res)
    output_lower_f_index = int(freq_lower_boundary/freq_res)
    
    # whistler cuts
    spectrograms_boundaries = [] 
    _indices = output_utilities.probable_output(indices, threshold)
    for index in _indices:
        # find maximum time boundary
        output_index = index[0]
        output_upper_t_index, output_lower_t_index = output_index, output_index
        if output_index + int(time_upper_boundary/time_res) <= time_max_index:
            output_upper_t_index += int(time_upper_boundary/time_res)
        else:
            extra = abs(time_max_index - (output_index + int(time_upper_boundary/time_res)))
            output_upper_t_index = time_max_index
            output_lower_t_index -= (int(time_lower_boundary/time_res)+extra)
            spectrograms_boundaries.append([output_lower_f_index,output_upper_f_index,output_lower_t_index,output_upper_t_index])
            continue
        if output_index - int(time_lower_boundary/time_res) >= 0:
            output_lower_t_index -= int(time_lower_boundary/time_res)
        else:
            extra = abs(output_index - int(time_lower_boundary/time_res))
            output_upper_t_index += extra
            output_lower_t_index = 0
        spectrograms_boundaries.append([output_lower_f_index,output_upper_f_index,output_lower_t_index,output_upper_t_index])
    spectrograms_boundaries = np.array(spectrograms_boundaries)
    # noise cuts
    noise_cuts = []
    f_cut_length = None
    t_cut_length = None
    if spectrograms_boundaries.size!=0:
        # get cuts from freq_upper_boundary
        f_max = spectrograms_boundaries[:,1].max()
        f_cut_length = int((parameters.freq_upper_boundary-parameters.freq_lower_boundary)/freq_res)
        t_cut_length = int((parameters.time_upper_boundary+parameters.time_lower_boundary)/time_res)-1
        freq_lower_index = int(parameters.freq_lower_boundary/freq_res)
        freq_upper_index = int(parameters.freq_upper_boundary/freq_res)
        if freq_max_index-f_max >= f_cut_length:
            for i in range(0,time_max_index,t_cut_length):
                if time_max_index-i>=t_cut_length:
                    noise_cuts.append([f_max,f_max+f_cut_length,i,i+t_cut_length])

        # get cuts from time_lower_boundary
        t_lowest_index = spectrograms_boundaries[:,2].min()
        if t_lowest_index-t_cut_length > 0:
            for i in range(0, t_lowest_index, int(t_cut_length/5)):
                if t_lowest_index-i>= t_cut_length:
                    noise_cuts.append([f_max, f_max+f_cut_length,i,i+t_cut_length])
        # get cuts from time_upper_boundary
    noise_cuts = np.array(noise_cuts)
    return _indices, spectrogram, spectrograms_boundaries, noise_cuts, f_cut_length, t_cut_length

