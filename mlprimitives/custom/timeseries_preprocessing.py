import warnings

import numpy as np
import pandas as pd
from scipy.fftpack import fft, rfft, irfft, fftfreq
import scipy.signal as signal
import matplotlib.pyplot as plt

def rolling_window_sequences(X, index, window_size, target_size, target_column):
    """Create rolling window sequences out of timeseries data."""
    out_X = list()
    out_y = list()
    X_index = list()
    y_index = list()

    target = X[:, target_column]

    for start in range(len(X) - window_size - target_size + 1):
        end = start + window_size
        out_X.append(X[start:end])
        out_y.append(target[end:end + target_size])
        X_index.append(index[start])
        y_index.append(index[end])

    return np.asarray(out_X), np.asarray(out_y), np.asarray(X_index), np.asarray(y_index)


_TIME_SEGMENTS_AVERAGE_DEPRECATION_WARNING = (
    "mlprimitives.custom.timeseries_preprocessing.time_segments_average "
    "is deprecated and will be removed in a future version. Please use "
    "mlprimitives.custom.timeseries_preprocessing.time_segments_aggregate instead."
)

def time_segments_average(X, interval, time_column):
    """Compute average of values over fixed length time segments."""
    warnings.warn(_TIME_SEGMENTS_AVERAGE_DEPRECATION_WARNING, DeprecationWarning)

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.sort_values(time_column).set_index(time_column)

    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]

    values = list()
    index = list()
    while start_ts <= max_ts:
        end_ts = start_ts + interval
        subset = X.loc[start_ts:end_ts - 1]
        means = subset.mean(skipna=True).values
        values.append(means)
        index.append(start_ts)
        start_ts = end_ts
    return np.asarray(values), np.asarray(index)


def time_segments_aggregate(X, interval, time_column, method=['mean']):
    """Aggregate values over fixed length time segments."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.sort_values(time_column).set_index(time_column)

    if isinstance(method, str):
        method = [method]

    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]

    values = list()
    index = list()
    while start_ts <= max_ts:
        end_ts = start_ts + interval
        subset = X.loc[start_ts:end_ts - 1]
        aggregated = [
            getattr(subset, agg)(skipna=True).values
            for agg in method
        ]
        values.append(np.concatenate(aggregated))
        index.append(start_ts)
        start_ts = end_ts
    return np.asarray(values), np.asarray(index)


def fft_transform_2(X, interval, time_column):
    """Compute FFT of signal"""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    #print(fft(np.real(np.asarray(X)[:, 0]), axis=0))
    X = X.sort_values(time_column).set_index(time_column)
    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]
    index = list()
    values = []
    while start_ts <= max_ts:
        end_ts = start_ts + interval
        subset = np.asarray(X.loc[start_ts : end_ts - 1])
        #subset = X[(X['timestamp']>= start_ts) & (X['timestamp']<= end_ts)]
        #print(np.asarray(subset))
        fft_aggregated = fft(np.asarray(subset)[:, 0], axis=0)
        values.append(fft_aggregated)
        index.append(start_ts)
        start_ts = end_ts
    #np.savetxt('test.txt', np.real(values))
    return np.real(np.asarray(values)), np.asarray(index)



def butterworth_filter(X, interval, time_column):

    """
    # Use fft and cut high frequencies, then inverse fft; not the best idea, rather use Window function
    X = X.sort_values(time_column).set_index(time_column)
    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]
    W = fftfreq(len(X.index))
    f_signal = rfft(np.asarray(X)[:,0])
    index = list()
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W>2)] = 0
    X_trans = irfft(cut_f_signal)
    print(X_trans)
    return np.reshape(np.asarray(X_trans), (-1, 1)), np.asarray(X.index.values)
    """
    X = X.sort_values(time_column).set_index(time_column)
    print(np.asarray(X)[:,0])
    N = 5
    Wn = 0.05
    B, A = signal.butter(N, Wn, output='ba')
    smooth_data = signal.filtfilt(B, A, np.asarray(X)[:,0])
    return np.reshape(np.asarray(smooth_data), (-1,1)), np.asarray(X.index.values)



def maximum_amplitude_fft(X, interval, index):
    """Compute maximum amplitude of ffts of last intervals """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    index_2 = list()
    interval = 5
    max_ampl = list()
    for start_ts in range(len(X) - interval):
        end_ts = start_ts + interval
        subset = np.asarray(X[start_ts:end_ts])
        fft_aggregated = rfft(subset)
        max_ampl.append(max(fft_aggregated))
        index_2.append(index[start_ts])

    y_hat = list()
    y_true = list()
    X_index = list()
    y_index = list()
    window_size = 3
    for start in range(len(max_ampl) - window_size):
        end = start + window_size
        y_hat.append(pd.DataFrame(max_ampl[start:end]).ewm(span=3).mean().mean())
        y_true.append(max_ampl[end])
        X_index.append(index_2[start])
        y_index.append(index_2[end])

    return np.asarray(y_hat), np.asarray(y_true), np.asarray(X_index), np.asarray(y_index)
