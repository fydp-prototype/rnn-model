from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.io.wavfile import read

rate1, data1 = read("man1_nb.wav")
rate2, data2 = read("man_lowpass.wav")
wide_arr = np.array(data1, dtype=float)
narrow_arr = np.array(data2, dtype=float)

print(rate1)
print(np.shape(data1))
print(rate2)

# fs = 10e3
# N = 1e5
# amp = 2 * np.sqrt(2)
# noise_power = 0.001 * fs / 2
# time = np.arange(N) / fs
# freq = np.linspace(1e3, 2e3, N)
# x = amp * np.sin(2*np.pi*freq*time)
# x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
#
# f, t, Sxx = spectrogram(x, fs, nperseg=512)
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# f0 = plt.figure()
# plt.plot(wide_arr)


# f4 = plt.figure()
plt.plot(narrow_arr)


# x = np.fft.fft(wide_arr)
# plt.plot(x)

f1 = plt.figure()
f, t, Sxx = spectrogram(wide_arr, rate1, nperseg=1024)
axes = plt.gca()
axes.set_xlim([0, 8])
axes.set_ylim([0, 8000])
plt.pcolormesh(t, f, 20*np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()


f2 = plt.figure()
f, t, Sxx2 = spectrogram(narrow_arr, rate2, nperseg=1024)
axes = plt.gca()
axes.set_xlim([0, 8])
axes.set_ylim([0, 8000])
plt.pcolormesh(t, f, 20*np.log10(Sxx2))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()
