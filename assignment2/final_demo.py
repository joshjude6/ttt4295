import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import sounddevice as sd
from hrir import hrir
from hrtf1 import hrtf1
from hrtfiir import hrtiir
from hrir_gen import hrir_gen

fs = 44100
c_air = 343
h_radius = 0.09
angles = [-90, -60, -30, 0, 30, 60, 90]
burst_dur = 0.25
gap_dur = 0.05
play_demo = True


def pink_noise(N):
    # 1/f pink noise using frequency shaping
    X = np.random.randn(N//2 + 1) + 1j*np.random.randn(N//2 + 1)
    f = np.linspace(1, N//2 + 1, N//2 + 1)
    X /= np.sqrt(f)
    x = np.fft.irfft(X)
    x /= np.max(np.abs(x))
    return x


def task5_demo():
    N_burst = int(burst_dur * fs)
    N_gap = int(gap_dur * fs)
    silence = np.zeros(N_gap)

    L_total = np.array([])
    R_total = np.array([])

    for ang in angles:
        burst = pink_noise(N_burst)

        # get hrirs for this angle
        hL, hR = hrir_gen(ang, h_radius, fs, c_air)

        # filter through hrirs (iir + itd)
        yL = lfilter(hL, [1.0], burst)
        yR = lfilter(hR, [1.0], burst)

        # append with gap
        L_total = np.concatenate((L_total, yL, silence))
        R_total = np.concatenate((R_total, yR, silence))

    maxamp = max(np.max(np.abs(L_total)), np.max(np.abs(R_total)))
    L_total /= maxamp
    R_total /= maxamp

    stereo = np.column_stack((L_total, R_total))
    t = np.arange(len(L_total)) / fs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1.plot(t, L_total, color='tab:blue')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Left ear')
    ax1.grid(True)
    
    ax2.plot(t, R_total, color='tab:orange')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Right ear')
    ax2.grid(True)
    
    fig.suptitle('Task 5 – Moving sound demo')
    plt.tight_layout()
    plt.show()

    if play_demo:
        print("playing demo - listen with headphones...")
        sd.play(stereo, fs)
        sd.wait()

    return stereo


if __name__ == "__main__":
    stereo_out = task5_demo()
