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

    # for start in range(len(X) - window_size - target_size + 1):
    #     end = start + window_size
    #     out_X.append(X[start:end])
    #     out_y.append(target[end:end + target_size])
    #     X_index.append(index[start])
    #     y_index.append(index[end])

    for start in range(len(X) - window_size - target_size + 1):
        end = start + window_size
        X_index.append(index[start])
        y_index.append(index[end])
    start = 0
    while start < len(X) - window_size - target_size + 1:
        end = start + window_size
        out_X.append(X[start:end])
        out_y.append(target[end:end + target_size])
        start = start + target_size

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


def fft_transform(X, interval, time_column):
    """Compute FFT of signal"""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    X = X.sort_values(time_column).set_index(time_column)
    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]
    index = list()
    values = []
    while start_ts <= max_ts:
        end_ts = start_ts + interval
        subset = np.asarray(X.loc[start_ts : end_ts - 1])
        fft_aggregated = fft(np.asarray(subset)[:, 0], axis=0)
        values.append(fft_aggregated)
        index.append(start_ts)
        start_ts = end_ts
    return np.real(np.asarray(values)), np.asarray(index)



def butterworth_filter(X, time_column, N, Wn):
    """Apply the Butterworth filter to the signal"""
    X = X.sort_values(time_column).set_index(time_column)
    B, A = signal.butter(N, Wn, output='ba')
    filtered_data = signal.filtfilt(B, A, np.asarray(X)[:,0])
    return np.reshape(np.asarray(filtered_data), (-1,1)), np.asarray(X.index.values)



def maximum_amplitude_fft(X, index, interval, window_size):
    """Compute maximum amplitude of ffts of last intervals """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    index_fft = list()
    max_ampl = list()
    for start_ts in range(len(X) - interval):
        end_ts = start_ts + interval
        subset = np.asarray(X[start_ts:end_ts])
        fft_aggregated = rfft(subset)
        max_ampl.append(max(fft_aggregated))
        index_fft.append(index[start_ts])

    y_hat = list()
    y_true = list()
    X_index = list()
    y_index = list()
    for start in range(len(max_ampl) - window_size):
        end = start + window_size
        y_hat.append(pd.DataFrame(max_ampl[start:end]).ewm(span=3).mean().mean())
        y_true.append(max_ampl[end])
        X_index.append(index_fft[start])
        y_index.append(index_fft[end])

    return np.asarray(y_hat), np.asarray(y_true), np.asarray(X_index), np.asarray(y_index)
