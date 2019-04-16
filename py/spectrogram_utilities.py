
# coding: utf-8

# In[1]:


import os
import sys
sys.path.insert(0,'../py')
sys.path.insert(0, '..')
import parameters
import numpy as np
import pandas as pd
import datetime
from scipy import signal as signal
import matplotlib.pyplot as plt


# In[3]:


def frread(fname=None):
    """ This is a rough translation of frread.m from J. Lichtenberger for the
    stereo=True case, i.e. we assume orthogonal loop antenna.
    inputs
        fname (string): File name path to the .vr2 file to load
    outputs
        wh (ndarray): 2xN array with the two traces in the first and second rows.
    """
    # open file for reading
    fid = open(fname, 'rb')
    # get data from file - 16-bit signed integers
    dat = np.fromfile(fid, dtype=np.int16)
    # length of one frame
    frLen = 4103  ## not sure how this is determined
    # number of frames to read
    nFrameRead = len(dat) / frLen
    # data length of frame
    adatlen = 2048
    # length of data set
    N = int(nFrameRead * adatlen)
    wh = np.zeros((N, 2), dtype=float)
    # for every frame
    for i in np.arange(0, nFrameRead, dtype=int):
        # indices for first component
        i1 = np.arange(7 + i * frLen, (i + 1) * frLen, 2, dtype=int)
        # indices for second component
        i2 = np.arange(8 + i * frLen, (i + 1) * frLen + 0, 2, dtype=int)
        ii = np.arange(i * adatlen, (i + 1) * adatlen, dtype=int)
        wh[ii, 0] = dat[i1]
        wh[ii, 1] = dat[i2]
#     print(len(np.arange(0, nFrameRead, dtype=int)))
    return wh

def vr2_to_panda(dir_name,fname, site):
    """Extract the data from a file a store it as a Panda DataFrame
    inputs
        fname    file name
        site     name of the site where data was collected
    outputs 
        whdf     dataframe containing the signal received by the NS and EW pointitng
                    orthogonal loop antennas
        fs       sampling frequency
        t0       start time
        t1       end time
    """
    # read vr2 file
    wh = frread(os.path.join(dir_name,fname))
    
    # CONSTANTS
    # Sampling frequency (20kHz for SANAE, 40kHz for MARION )
    fs = 2e4 if site=="sanae" else 4e4
    # time step in microseconds (for dataframe index)
    dt = 1e6 / fs

    # Set the date/time format in the filename
    # dtFormat = '%Y-%m-%dUT%H_%M_%S.%f'
    dtFormat = '%Y-%m-%dUT%H:%M:%S.%f'

    # Set up pandas dataframe
    # Start time
    t0 = pd.datetime.strptime(fname[0:27], dtFormat)
    # Number of samples
    Nsamples = len(wh[:, 0])
    # End time
    t1 = t0 + datetime.timedelta(0, 0, Nsamples * dt)
    # Create index
    tindex = pd.date_range(start=t0, periods=Nsamples, freq='50U') # freq = 50us

    # Create pandas data frame from wh
    whdf = pd.DataFrame(index=tindex, data=wh[:, 0], columns=['X'])
    whdf['Y'] = wh[:, 1]
    # The 'X' and 'Y' columns are the signal received by the North/South and
    # East/West pointing orthogonal loop antennas used at Marion and SANAE
    
    return whdf, fs

def spectrogram_gen(data, fs, scaling=parameters.scaling):
    """Compute spectrogram from vr2 data collected
    inputs
        data       Pandas DataFrame of the vr2 data
        fs         Sampling frequency
    outputs
        data_info  dictionary of the frequencies, time, and spectrum of the sprectrogram
    """
    frequencies, times, spectrogram = signal.spectrogram(
        data.X.values, 
        fs=fs, 
        nperseg = parameters.nperseg,
        noverlap=parameters.noverlap,
        nfft=parameters.nfft, 
        detrend=parameters.detrend, 
        scaling=scaling,
        mode = parameters.mode)
    return frequencies, times, np.log10(spectrogram)

def reshape_spectrogram(f, t, s):
    '''Reshape the frequency, time, and spectrogram data into one 2D array
    params:
        time
        frequency
        spectrogram
    return:
        2D array
    '''
    f = np.asarray(f)
    t = np.asarray(t)
    s = np.asarray(s)
    _t = np.concatenate(([0],t))
    _s = np.concatenate((f[np.newaxis].T,s), axis=1)
    sft = np.vstack((_t,_s))
    return sft

def spectrogram_from_vr2(dir_name,fname, site,
                        nperseg = parameters.nperseg,
                        noverlap=parameters.noverlap,
                        nfft=parameters.nfft, 
                        detrend=parameters.detrend,
                        scaling=parameters.scaling,
                        mode = parameters.mode):
    whdf, fs = vr2_to_panda(dir_name,fname, site)
    freq, time, spec = signal.spectrogram(
        whdf.X.values, 
        fs=fs, 
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft, 
        detrend=detrend, 
        scaling=scaling,
        mode=mode)
    return freq, time, np.log10(spec)
    

def spectrogram_plot(spectrogram,size=None, x_axis=None, y_axis=None, x_label=None, y_label=None, scaling=parameters.scaling, cmap='jet' ):
    if size:
        fig = plt.figure(figsize=(10,3))
    else:
        fig = plt.figure()
    if x_axis is not None and y_axis is not None:
        img = plt.pcolormesh(y_axis, x_axis, spectrogram, cmap=cmap)
    else:
        img = plt.pcolormesh(spectrogram, cmap=cmap)
    fig.colorbar(mappable=img, label=str(scaling))
    if x_label and y_label :
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (kHz)')
    plt.show()
    return spectrogram
    
def spectrogram_full_plot(dir_name,fname, site, cmap='jet',
                        nperseg = parameters.nperseg,
                        noverlap=parameters.noverlap,
                        nfft=parameters.nfft, 
                        detrend=parameters.detrend,
                        scaling=parameters.scaling,
                        mode = parameters.mode):
    freq, time, spec = spectrogram_from_vr2(dir_name,fname, site,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft, 
        detrend=detrend, 
        scaling=scaling,
        mode=mode)
    spectrogram_plot(spec, size=(16,5), x_axis=freq/10**3, y_axis=time, x_label='Time (s)', y_label='Frequency (kHz)', scaling=scaling+' [dB]', cmap=cmap)
    return freq, time, spec

