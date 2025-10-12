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

def plot_spectrum(filename, file_number, n_fft=16384, fmin=20.0, threshold=0.1):
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
    
    # Convert to dB
    mag_db = 20 * np.log10(mag + 1e-12)  # Add small value to avoid log(0)

    peaks, _ = find_peaks(mag, height=0.1*np.max(mag))
    
    # Harmonic analysis
    strong = [p for p in peaks if mag[p] >= threshold * np.max(mag)]
    strong_sorted = sorted(strong, key=lambda i: freqs[i])
    strong_sorted = [i for i in strong_sorted if freqs[i] >= fmin]
    
    if strong_sorted:
        max_peak_idx = max(strong_sorted, key=lambda i: mag[i])
        f0 = freqs[max_peak_idx]
        tolerance_hz = rate/n_fft
        
        groupA, groupB = [], []
        for idx in strong_sorted:
            f = freqs[idx]
            if f < f0 - tolerance_hz:
                continue
            k = int(round(f / f0))
            if k < 1:
                groupB.append(idx)
                continue
            f_ideal = k * f0
            dev_hz = f - f_ideal
            if abs(dev_hz) <= tolerance_hz:
                groupA.append(idx)
            else:
                groupB.append(idx)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mag_db, 'b-', linewidth=0.8)
    
    if strong_sorted:
        plt.scatter(freqs[groupA], mag_db[groupA], color='green', s=50, label='Group A (Harmonics)')
        plt.scatter(freqs[groupB], mag_db[groupB], color='red', s=50, label='Group B (Non-harmonic)')
    else:
        plt.scatter(freqs[peaks], mag_db[peaks], color='red', s=50, label='Peaks')

    for peak in peaks[np.argsort(mag[peaks])[-5:]]:
        plt.annotate(f'{freqs[peak]:.03f} Hz', 
                    (freqs[peak], mag_db[peak]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left')
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.title(f'Spectrum for audio recording {file_number}')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 3000)
    plt.show()

def analyze_file(file_number):
    filename = f"/Users/joshjude/Documents/Git/ttt4295/assignment1/music_box_tones_k/pink-panther_{file_number:03d}.wav"
    plot_waveform(filename, file_number)
    plot_spectrum(filename, file_number)

analyze_file(32)