# IMPORTS
import numpy as np
import pandas as pd
import datetime
from scipy import signal as signal
import matplotlib.pyplot as plt


# CONSTANTS


def main():
    fname = "..\\Data\\sanae\\2012-02-02UT20_58_32.84943133.sanae.vr2"

    # read vr2 file
    wh = frread(fname)

    # set constants
    ## sampling frequency (20kHz for SANAE, 40kHz for MARION)
    #fs = 4e4
    fs = 2e4

    ## time step in microseconds (for dataframe index)
    dt = 1e6 / fs

    ## set the date/time format in the filename
    # dtFormat = '%Y-%m-%dUT%H_%M_%S.%f'
    dtFormat = '%Y-%m-%dUT%H:%M:%S.%f'

    # set up pandas dataframe
    ## start time
    t0 = pd.datetime.strptime(fname[0:27], dtFormat)
    ## number of samples
    Nsamples = len(wh[:, 0])
    ## end time
    t1 = t0 + datetime.timedelta(0, 0, Nsamples * dt)
    ## create index
    tindex = pd.date_range(start=t0, periods=Nsamples, freq='50U')

    ## create pandas data frame from wh
    whdf = pd.DataFrame(index=tindex, data=wh[:, 0], columns=['X'])
    whdf['Y'] = wh[:, 1]
    ### the 'X' and 'Y' columns are the signal received by the North/South and
    ### East/West pointing orthogonal loop antennas used at Marion and SANAE

    ## make a nice string
    t0str = pd.datetime.strftime(t0, '%Y-%m-%d %H:%M:%S.%f')
    t1str = pd.datetime.strftime(t1, '%Y-%m-%d %H:%M:%S.%f')
    print('%s -- %s' % (t0str, t1str))

    ### get spectrogram
    # f,t,sxx = signal.spectrogram(whdf.X.values,fs=fs,\
    # scaling='spectrum',\
    # mode='magnitude',\
    # detrend='linear')

    # f,t,sxx = signal.spectrogram(whdf.X.values,fs=fs,\
    # scaling='spectrum',\
    # mode='magnitude',\
    # detrend='linear')

    ## plot spectrogram
    # plt.figure()
    ##plt.pcolormesh(t,f,sxx)
    # plt.pcolormesh(t,f,np.log10(sxx))
    # plt.ylabel('Freq [Hz]')
    # plt.xlabel('Time [s] since %s' % t0str[10:])
    # plt.title(t0str[0:10])
    # plt.colorbar()

    # plot spectrogram using matplotlib
    fig = plt.figure()
    ax1 = plt.gca()
    s, f, t, img = ax1.specgram(whdf.X.values, Fs=fs, detrend='linear', NFFT=512, noverlap=64, scale='dB',
                                scale_by_freq=False)
    fig.colorbar(mappable=img, label='Power [dB]')

    # plot timeseries
    plt.figure()
    whdf.X.plot(label='X signal (NS)')
    whdf.Y.plot(label='Y signal (EW)')
    plt.title(t0str[0:10])
    plt.xlabel('Time UT')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid()

    return whdf


def frread(fname=None):
    """
    This is a rough translation of frread.m from J. Lichtenberger for the
    STEREO=TRUE case, i.e. we assume orthogonal loop antenna.

    INPUTS
        fname (string)	File name path to the .vr2 file to load

    OUTPUTS
        wh (ndarray)	2xN array with the two traces in the first and second rows.
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
        print('frame %d' % i)

        # indices for first component
        i1 = np.arange(7 + i * frLen, (i + 1) * frLen, 2, dtype=int)
        # indices for second component
        i2 = np.arange(8 + i * frLen, (i + 1) * frLen + 0, 2, dtype=int)

        ii = np.arange(i * adatlen, (i + 1) * adatlen, dtype=int)

        wh[ii, 0] = dat[i1]
        wh[ii, 1] = dat[i2]

    return wh


if __name__ == "__main__":
    main()
