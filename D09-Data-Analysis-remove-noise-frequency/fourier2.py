import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from get_data import Data
from scipy.signal import find_peaks
import cowsay
# Load the data
import os
mat = []
for i in range(1, 7):
    line = []
    for j in range(1, 7):
        line.append(Data(i, j))
    mat.append(line)

def plot_and_save(subject, trial, run):
    ft = mat[subject][trial].ft
    u = mat[subject][trial].u
    u = np.transpose(u)
    u = u[run]
    t = mat[subject][trial].t[0]
    x = mat[subject][trial].x
    x = np.transpose(x)
    x = x[run]
    print(cowsay.cow("fourier transform"))

    # Compute the FFT of the signal
    fft_x = np.fft.fft(x)
    freq = np.fft.fftfreq(len(t), t[1] - t[0])  # Compute the frequencies for the x-axis

    # Identify the noise frequencies and set them to zero
    # Here, we consider frequencies with small magnitudes as noise
    threshold = 0.1 # This value may need to be adjusted
    fft_x_clean = fft_x.copy()
    print(np.abs(fft_x_clean))
    fft_x_clean[np.abs(fft_x) < threshold] = 0

    # Compute the inverse FFT of the cleaned signal
    x_clean = np.fft.ifft(fft_x_clean)

    # Plot the original, Fourier transformed, and cleaned signals
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(freq, x, label='Original')
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.semilogy(freq, np.abs(fft_x), label='FFT')
    plt.title('Fourier Transform')
    plt.xlabel('Frequency')
    plt.xlim(0, 5)
    plt.ylabel('Magnitude')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(freq, x_clean, label='Cleaned')
    plt.title('Noise-cancelled Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_and_save(1, 3, 1)