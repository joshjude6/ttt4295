import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.io import wavfile

def plot_waveform(filename, file_number, n_fft=16384):
    rate, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data[:, 0]
    
    time = np.arange(len(data)) / rate
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, data, 'b-', linewidth=0.8)
    
    if len(data) >= n_fft:
        fft_time_end = time[n_fft-1]
        plt.axvspan(0, fft_time_end, alpha=0.3, color='red', label=f'FFT window')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform for audio recording {file_number}')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_spectrum(filename, file_number, n_fft=16384):
    rate, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data[:, 0]

    if len(data) < n_fft:
        x = np.pad(data, (0, n_fft - len(data)))
    else:
        x = data[:n_fft]

    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(n_fft, 1/rate)[:n_fft//2]
    mag = np.abs(X[:n_fft//2])

    peaks, _ = find_peaks(mag, height=0.1*np.max(mag))

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mag, 'b-', linewidth=0.8)
    plt.scatter(freqs[peaks], mag[peaks], color='red', s=50, label='Peaks')

    for peak in peaks[np.argsort(mag[peaks])[-5:]]:
        plt.annotate(f'{freqs[peak]:.03f} Hz', 
                    (freqs[peak], mag[peak]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left')
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title(f'Spectrum for audio recording {file_number}')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 2000)
    plt.show()

def analyze_file(file_number):
    filename = f"/Users/joshjude/Documents/Git/ttt4295/assignment1/music_box_tones/tone_music_box_full_audio_{file_number:03d}.wav"
    plot_waveform(filename, file_number)
    plot_spectrum(filename, file_number)

analyze_file(29)