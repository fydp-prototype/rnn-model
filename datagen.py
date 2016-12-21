# import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, irfft
from scipy.io.wavfile import read, write
from scipy.signal import butter, lfilter


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=6):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def write_arr_to_wav(arr, filename, rate):
    write(filename, rate, np.array(arr).astype('int16'))


def convert_to_freq_blocks(time_blocks):
    return [rfft(b) for b in time_blocks]


def convert_to_blocks(time_signal, block_size):
    time_signal = time_signal[0:len(time_signal) - len(time_signal) % block_size]
    # if the block_size is 1024, then there will be 512+1 frequency buckets
    # each frequency is 16000*i/1024 (evenly distributed from 0 to 8000Hz which is the nyquist limit)
    return np.array_split(time_signal, len(time_signal) // block_size)


def convert_time_signal_to_freq_blocks(time_signal):
    time_blocks = convert_to_blocks(time_signal, 128)
    freq_blocks = convert_to_freq_blocks(time_blocks)
    return freq_blocks


def flatten(l):
    return [item for sublist in l for item in sublist]


def convert_freq_to_time(freq_block):
    return irfft(freq_block)


def convert_freq_blocks_to_time(freq_blocks):
    return flatten([convert_freq_to_time(b) for b in freq_blocks])


def read_wav(filename):
    rate, data = read(filename)
    time_signal = np.array(data, dtype=float)
    return time_signal, rate


def get_training_input():
    time_signal, rate = read_wav("man1_wb.wav")
    filtered_time_signal = lowpass_filter(time_signal, cutoff=4000, fs=float(rate))
    filtered_freq_blocks = convert_time_signal_to_freq_blocks(
        filtered_time_signal)  # converts each block into list of complex numbers
    nums = [np.concatenate((b.real, b.imag)) for b in filtered_freq_blocks]
    return nums


def get_training_output():
    time_signal, rate = read_wav("man1_wb.wav")
    freq_blocks = convert_time_signal_to_freq_blocks(time_signal)  # converts each block into list of complex numbers
    nums = [np.concatenate((b.real, b.imag)) for b in freq_blocks]
    return nums


def get_freq_blocks_from_vector(vector):
    ret = []
    for i in range(len(vector) / 2):
        ret.append(vector[i] + vector[i + len(vector) / 2] * 1j)
    return ret


input = get_training_input()
print(len(input))
freq_blocks = [get_freq_blocks_from_vector(vec) for vec in input]
time = convert_freq_blocks_to_time(freq_blocks)
write_arr_to_wav(time, "test.wav", 16000)

x = get_training_output()
print(len(x))
