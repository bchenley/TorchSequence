from scipy import signal as sp

def butter(x, critical_frequency, butter_type = 'low', filter_order = 3, sampling_rate = 1):
    '''
    Applies a Butterworth filter to the input signal.

    Args:
        x: The input signal.
        critical_frequency: The critical frequency of the filter.
        butter_type: The type of Butterworth filter to apply.
        filter_order: The order of the filter.
        sampling_rate: The sampling rate of the input signal.

    Returns:
        y: The output signal after applying the Butterworth filter.
    '''
    b, a = sp.butter(N=filter_order,
                     Wn=critical_frequency / (sampling_rate / 2),
                     btype=butter_type,
                     output='ba')

    y = sp.filtfilt(b, a, x, axis=0)
    return y
