import numpy as np


def get_idx(array, value):
    """
    Gives the index of the element in the array closest to the value.

    :param array: sorted array, e.g. time or freq vector.
    :param value: value to look for
    :return: index of closest element
    """
    from bisect import bisect

    idx = bisect(array, value) - 1
    if abs(array[idx] - value) < abs(array[idx + 1] - value):
        return idx
    else:
        return idx + 1


def element_wise_mean(dicts, axis=(0, 1)):
    import file_logic as file

    data_mean = None
    for file_no, file_dict in enumerate(dicts):
        data = file.load_file(file_dict, axis)
        if file_no is 0:
            data_mean = data
        else:
            data_mean = (data + file_no * data_mean) / (file_no + 1)
    return data_mean


def get_freq_range_slice(freq, freq_center, freq_radius):
    """

    :param freq: array of frequency values
    :param freq_center: center of the slice
    :param freq_radius: value for distance of freq_center to start and stop of slice
    :return: frequency range slice
    """
    freq_start = get_idx(freq, freq_center - freq_radius)
    freq_stop = get_idx(freq, freq_center + freq_radius)
    freq_center = get_idx(freq, freq_center)

    if freq_stop is freq_start or freq_stop is freq_center:
        freq_stop += 1
    return slice(freq_start, freq_stop)


def data_filter(data, filter_freq, sampling_rate, filter_type='high'):
    """Filters data with scypi butter filter.

    :param data:
    :param filter_freq:
    :param sampling_rate:
    :param filter_type:
    :return: filtered data
    """
    from scipy.signal import buttord, butter, sosfiltfilt

    if filter_type == 'high' and isinstance(filter_freq, (int, float)):
        wp = float(filter_freq)
        ws = 0.5 * wp
    elif filter_type == 'low' and isinstance(filter_freq, (int, float)):
        wp = float(filter_freq)
        ws = 1.5 * wp
    elif filter_type == 'band' and isinstance(filter_freq, (list, tuple)):
        wp = [float(filter_freq[0]), float(filter_freq[1])]
        ws = [0.5 * wp[0], 1.5 * wp[1]]
    else:
        raise ValueError("'%s' is an invalid filter type for this function. Use 'high', 'low' or 'band'" % filter_type)

    order, wn = buttord(wp, ws, 1, 30, fs=sampling_rate)
    filter_parameters = butter(N=order, Wn=wn, btype=filter_type, output='sos', fs=sampling_rate)
    filtered_data = sosfiltfilt(filter_parameters, np.array(data), padtype='even', padlen=None)
    return filtered_data


def filter_and_save(filter_freq=20, filter_type='high', **file_search_kwargs):
    import file_logic as file

    def do():
        for file_dict in raw_data_dicts:
            time_data, ampl_data = file.load_file(file_dict)

            sampling_rate = get_data_sampling_rate(time_data)
            ampl_data = data_filter(ampl_data, filter_freq, sampling_rate, filter_type)

            file_name = file.create_filename(**dict(file_dict,
                                                    raw_data=False,
                                                    filtering=filter_type,
                                                    filter_freq=filter_freq))
            file.save_file(time_data, ampl_data, file_name=file_name)

    raw_data_dicts = file.file_search(**dict(file_search_kwargs, raw_data=True))
    filtered_data_dicts = file.file_search(**dict(file_search_kwargs, raw_data=False))

    if not filtered_data_dicts and raw_data_dicts:
        do()
    elif 'filter_freq' in filtered_data_dicts[0].keys():
        if input(
                "Data is already filtered with {}Hz. "
                "Should files be filtered again with {}Hz? y/n".format(
                    filtered_data_dicts[0]["filter_freq"],
                    filter_freq)) is 'y':
            do()


def cut_and_save(prominence=0.0018, **file_search_kwargs):
    import file_logic as file

    def do():
        len_list = []
        peak_list = []
        for file_dict in filtered_data_dicts:
            file_name = file.create_filename(**file_dict)
            time_data, ampl_data = file.load_file(file_name)

            # peak = find_peaks(-ampl_data, [time_data < 0.15, time_data > 0.05], prominence=prominence)[0][0]
            peak = find_argmax(-ampl_data, [time_data < 0.15, time_data > 0.05], prominence)

            len_list.append(len(time_data[peak:]))
            peak_list.append(peak)

        min_len = min(len_list)
        min_peak = min(peak_list)

        print("time_len and peaks found, min len:", min_len)

        for file_dict, peak in zip(filtered_data_dicts, peak_list):
            file_name = file.create_filename(**file_dict)
            time_data, ampl_data = file.load_file(file_name)

            time_data -= time_data[peak - min_peak]
            time_data = time_data[peak - min_peak:peak + min_len]
            ampl_data = ampl_data[peak - min_peak:peak + min_len]

            new_file_name = file.create_filename(**dict(file_dict, is_cut=True))
            file.save_file(time_data, ampl_data, file_name=new_file_name)

        print("Cutting done, all data now has len =", min_len)

    filtered_data_dicts = file.file_search(**dict(file_search_kwargs, raw_data=False, is_cut=False))

    if filtered_data_dicts:
        do()
    else:
        print('no filtered data found')


def filter_cut_and_save(filter_freq=20, filter_type='high', prominence=0.0018, **file_search_kwargs):
    import file_logic as file

    def do():
        len_list = []
        peak_list = []
        for file_dict in raw_data_dicts:
            file_name = file.create_filename(**file_dict)
            time_data, ampl_data = file.load_file(file_name)

            sampling_rate = get_data_sampling_rate(time_data)
            ampl_data = data_filter(ampl_data, filter_freq, sampling_rate, filter_type)

            peak = find_argmax(-ampl_data, [time_data < 0.15, time_data > 0.05], prominence)

            len_list.append(len(time_data[peak:]))
            peak_list.append(peak)

            file_name = file.create_filename(**dict(file_dict,
                                                    raw_data=False,
                                                    filtering=filter_type,
                                                    filter_freq=filter_freq,
                                                    is_cut=False))
            file.save_file(time_data, ampl_data, file_name=file_name)

        min_len = min(len_list)
        min_peak = min(peak_list)

        print("data filtered and time_len and peaks found, min len:", min_len)

        new_filtered_data_dicts = file.file_search(**dict(file_search_kwargs,
                                                          raw_data=False,
                                                          filtering=filter_type,
                                                          filter_freq=filter_freq,
                                                          is_cut=False))

        for file_dict, peak in zip(new_filtered_data_dicts, peak_list):
            file_name = file.create_filename(**file_dict)
            time_data, ampl_data = file.load_file(file_name)

            time_data -= time_data[peak - min_peak]
            time_data = time_data[peak - min_peak:peak + min_len]
            ampl_data = ampl_data[peak - min_peak:peak + min_len]

            new_file_name = file.create_filename(**dict(file_dict, is_cut=True))
            file.save_file(time_data, ampl_data, file_name=new_file_name)

        print("Cutting done, all data now has len =", min_len)

    raw_data_dicts = file.file_search(**dict(file_search_kwargs, raw_data=True))
    filtered_data_dicts = file.file_search(**dict(file_search_kwargs, raw_data=False, is_cut=False))

    if not filtered_data_dicts and raw_data_dicts:
        do()
    elif 'filter_freq' in filtered_data_dicts[0].keys():
        if input(
                "Data is already filtered with {}Hz. "
                "Should files be filtered again with {}Hz? y/n".format(
                    filtered_data_dicts[0]["filter_freq"],
                    filter_freq)) is 'y':
            do()


def harm_save(f0=55, **search_kwargs):
    import file_logic as file

    file_dicts = file.file_search(**search_kwargs)

    for file_dict in file_dicts:

        time_data, ampl_data = file.load_file(file_dict)

        sampling_rate = get_data_sampling_rate(time_data)

        # FFT
        time_cut = 5
        freq_data, fft_data = fft(time_data[time_data < time_cut], ampl_data[time_data < time_cut], sampling_rate)

        peaks_total = 8
        freq_range = [f0-1, f0*peaks_total+1]
        freq_peaks = fft_peaks(fft_freq=freq_data,
                               fft_ampl=fft_data,
                               freq_range=freq_range,
                               number_of_peaks=peaks_total,
                               distance=f0-2)[0]

        # STFT
        nperseg = 2 ** 12 + 2 ** 11
        ngap = 2 ** 6 + 2 ** 5

        f, t, Zxx = stft_func(ampl_data, sampling_rate,
                              nperseg=nperseg, ngap=ngap)
        freq_range = range_and(f > f0 - 10, f < freq_peaks[-1] + 10)
        Zxx = Zxx[freq_range]
        f = f[freq_range]

        r = 3.5
        max_list = []
        for overtone_freq in freq_peaks:
            sl = get_freq_range_slice(f, overtone_freq, r)
            max_list.append(np.amax(Zxx[sl], axis=0))

        file.save_file(t, *max_list, file_name=dict(file_dict, description='harm'))


def Harm_mean_std_and_sum(harm_max=None, **search_kwargs):
    import file_logic as file

    file_dicts = file.file_search(**search_kwargs)

    data = file.load_file(file_dicts[0], axis=range(harm_max) if harm_max is not None else harm_max)
    time_data = data[0]

    harm_data_list = []

    # Harm-Daten laden
    axis = np.arange(1, len(data))
    for file_dict in file_dicts[:]:
        harm_data = file.load_file(file_dict, axis=axis)
        harm_data_list.append(harm_data)

    # Daten synchronisieren und auf summe normieren
    harm_sums = np.sum(harm_data_list, axis=1)
    sum_peaks = np.amax(harm_sums[:, time_data < 3], axis=1)
    sum_peaks_idx = np.argmax(harm_sums[:, time_data < 3], axis=1)

    data_len = len(time_data)
    min_len = data_len - max(sum_peaks_idx)
    min_peak_idx = min(sum_peaks_idx)
    shorter_len = min_len + min_peak_idx

    new_array = np.ones((len(file_dicts), len(data) - 1, shorter_len))
    for i in range(len(sum_peaks)):
        harm_data_list[i] = harm_data_list[i] / sum_peaks[i]
        np.copyto(new_array[i], harm_data_list[i][:, sum_peaks_idx[i] - min_peak_idx:sum_peaks_idx[i] + min_len])

    harm_data_list = new_array
    time_data = time_data[:shorter_len]
    del new_array

    # mean und std der Harmonischen
    harm_data_mean = np.mean(harm_data_list, axis=0)
    harm_data_std = np.std(harm_data_list, axis=0)

    # means und stds aufsummieren
    harm_data_mean_sum = np.sum(harm_data_mean, axis=0)
    harm_data_std_sum = np.sqrt(np.sum(np.square(harm_data_std), axis=0, dtype=np.float64))
    return time_data, harm_data_list, harm_data_mean, harm_data_std, harm_data_mean_sum, harm_data_std_sum


def harm_fit_and_shift(t, y, c_mean_cut, fit_range):
    y -= np.mean(y[t > c_mean_cut]) - np.std(y[t > c_mean_cut])

    popt = fit_exp_linear(t[fit_range], y[fit_range])
    popt = np.array([*popt, 0])
    return y, popt


def get_data_sampling_rate(time_data: np.ndarray) -> int:
    """
    Gives the sampling rate of the given data.

    :param time_data: time data
    :return: sample rate
    """
    sample_rates = []
    for i in range(len(time_data) - 1):
        sample_rates.append(1.0 / (time_data[i + 1] - time_data[i]))
    return int(np.mean(sample_rates))


def fft(time_data, ampl_data, sampling_rate=None):
    """fft of given data set

    :param time_data: list of time data
    :param ampl_data: list of amplitude data
    :param sampling_rate: sampling rate

    :return: lists of frequency and fft data
    """
    if sampling_rate is None:
        sampling_rate = get_data_sampling_rate(time_data)
    data_points_total = len(time_data)
    freq_vector = np.fft.rfftfreq(data_points_total, d=1 / sampling_rate)

    fft_vector = 2 / data_points_total * np.abs(np.fft.rfft(ampl_data))
    return np.array((freq_vector, fft_vector))


def fft_peaks(fft_freq, fft_ampl, freq_range=None, number_of_peaks=1, distance=300):
    """

    :param fft_freq: frequency vector of fft
    :param fft_ampl: amplitude vector of fft
    :param float or tuple or list or slice freq_range: frequency range in which to search for peaks in Hz
    :param int number_of_peaks: number of peaks to look for
    :param float distance: distance of peaks in Hz
    :return: array of peak_freq and peak_ampl with the length given bei number_of_peaks
    """
    from scipy.signal import find_peaks as fp
    from heapq import nlargest
    from numbers import Number

    distance = get_idx(fft_freq, distance) if distance is not None else None

    if freq_range is not None:
        if isinstance(freq_range, Number):
            freq_range = fft_freq <= freq_range
        elif isinstance(freq_range, (tuple, list)):
            freq_range = np.logical_and(fft_freq >= freq_range[0], fft_freq <= freq_range[1])
        elif isinstance(freq_range, slice):
            freq_range = freq_range
        else:
            raise ValueError('Unknown freq_range format')
    else:
        freq_range = slice(None)

    fft_freq, fft_ampl = fft_freq[freq_range], fft_ampl[freq_range]

    peaks, properties = fp(fft_ampl, height=0, prominence=0, distance=distance)
    prom = nlargest(number_of_peaks, properties["prominences"])[-1] if len(peaks) is not 0 else 0
    peaks, _ = fp(fft_ampl, height=0, prominence=prom, distance=distance)

    return fft_freq[peaks], fft_ampl[peaks]


def find_peaks(data, data_range=None, **kwargs):
    from scipy.signal import find_peaks

    mask = [True] * len(data)
    if isinstance(data_range, (tuple, list)):
        if len(data_range) is 2:
            mask = np.bitwise_and(*data_range)
        elif len(data_range) == len(data):
            mask = data_range
        else:
            raise ValueError('unknown data_range format')
    elif isinstance(data_range, slice):
        mask = data_range

    try:
        peak_shift = list(mask).index(True)
    # except ValueError:
    #     raise ValueError('data_range schränkt data zu sehr ein')
    except TypeError:
        peak_shift = mask.start

    peaks, properties = find_peaks(data[mask], **kwargs)

    peaks = [peak + peak_shift for peak in peaks]

    return peaks, properties


def stft_func(ampl_data, sampling_rate, nperseg=2 ** 15, ngap=None, window='hamm', **stft_kwargs):
    """
    Compute the Short Time Fourier Transform.

    :param window:
    :param ampl_data:
    :param sampling_rate:
    :param nperseg:
    :param ngap:
    :return: frequency and time vectors and amplitude matrix
    """
    from scipy.signal import stft
    if ngap is None:
        ngap = nperseg / 2
    f, t, Zxx = stft(ampl_data, sampling_rate,
                     nperseg=nperseg, noverlap=nperseg - ngap,
                     window=window, **stft_kwargs)
    return f, t, np.abs(Zxx)


def stft_plot(ampl_data, sampling_rate,
              nperseg=2 ** 15, ngap=None,
              ax=None, t_lim=None, f_lim=None, del_f=True,
              **stft_kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    show_cond = False
    if ax is None:
        show_cond = True
        _, ax = plt.subplots()

    if t_lim is not None:
        if isinstance(t_lim, int) or isinstance(t_lim, float):
            ax.set_xlim(right=t_lim)
        elif isinstance(t_lim, tuple):
            ax.set_xlim(t_lim)

    if f_lim is not None:
        if isinstance(f_lim, int) or isinstance(f_lim, float):
            ax.set_ylim(top=f_lim)
        elif isinstance(f_lim, tuple):
            ax.set_ylim(f_lim)

    f, t, Zxx = stft_func(ampl_data, sampling_rate, nperseg, ngap, **stft_kwargs)

    if del_f and (isinstance(f_lim, int) or isinstance(f_lim, float)):
        f = f[f < f_lim]
        Zxx = Zxx[:len(f)]

    ax.pcolormesh(t, f, Zxx, norm=LogNorm())

    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')

    if show_cond:
        plt.show()

    return f, t, Zxx


def range_and(x1, x2=None):
    """
    Takes two bool arrays of same length and gives back intersection

    :param x1: bool array
    :param x2: bool array, can be left out, then x1 will be returned
    :return:
    """
    if x2 is None:
        return x1
    else:
        return np.bitwise_and(x1, x2)


def range_cut(x, bound0, bound1=None):
    x_bools = [None, None]

    if bound1 is not None:
        if bound0 < bound1:
            bound0, bound1 = bound1, bound0
        check_list = [bound0, bound1]
    else:
        check_list = [bound0]

    check_list_names = ["upper_bound", "lower_bound"]
    for to_check_no in range(len(check_list)):
        if x[0] > check_list[to_check_no]:
            print("Warning: {} smaller than first value of x".format(check_list_names[to_check_no]))
        elif x[-1] < check_list[to_check_no]:
            print("Warning: {} bigger than last value of x".format(check_list_names[to_check_no]))

    if bound1 is None:
        x_bools[1] = None
    else:
        x_bools[1] = x > bound1

    x_bools[0] = x < bound0

    return range_and(*x_bools)


def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def exp_fit(xdata, ydata, start_region=0.8, stop_value=None, full=False):
    from scipy.optimize import curve_fit

    start_index = np.argmax(ydata[xdata < start_region])
    stop_index = get_idx(xdata, stop_value) if stop_value is not None else None
    popt, pcov = curve_fit(exp_func, xdata[start_index:stop_index], ydata[start_index:stop_index])
    return popt, pcov if full else popt


def fit_exp_linear(t, y, c=0):
    y = y - c
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, -K


def find_argmax(data, data_range, peak_treshold):
    mask = [True] * len(data)
    if isinstance(data_range, (tuple, list)):
        if len(data_range) is 2:
            mask = np.bitwise_and(*data_range)
        elif len(data_range) == len(data):
            mask = data_range
        else:
            raise ValueError('unknown data_range format')
    elif isinstance(data_range, slice):
        mask = data_range

    try:
        peak_shift = list(mask).index(True)
    except ValueError:
        raise ValueError('data_range schränkt data zu sehr ein')
    except TypeError:
        peak_shift = mask.start

    data_index_range = data > peak_treshold
    search_stop_idx = -1
    if not np.all(data_index_range):
        val_before = False
        for idx, bool_val in enumerate(data_index_range):
            if not bool_val and val_before:
                search_stop_idx = idx
                break
            val_before = bool_val
    elif np.any(data_index_range):
        raise ValueError("peak threshold too high")
    else:
        raise ValueError("peak threshold too low")

    return np.argmax(data[:search_stop_idx]) + peak_shift


def start_timer():
    from time import perf_counter
    return perf_counter()


def stop_timer(start_time, printing=True):
    from time import perf_counter
    stop_time = perf_counter()
    time_elapsed = stop_time - start_time
    if printing:
        print(time_elapsed)
    return time_elapsed
