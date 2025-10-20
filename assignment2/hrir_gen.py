import numpy as np
import matplotlib.pyplot as plt
from hrir import hrir
from hrtf1 import hrtf1
from hrtfiir import hrtiir
from scipy.signal import lfilter, freqz

def hrir_gen(inc_angle, h_radius, fs, c_air):
    BL, AL, BR, AR = hrtiir(inc_angle, h_radius, fs, c_air)

    impulse = np.zeros(512)
    impulse[0] = 1
    h_left = lfilter(BL, AL, impulse)
    h_right = lfilter(BR, AR, impulse)

    hL_delay, hR_delay = hrir(inc_angle, h_radius, fs, c_air)
    delayL = np.argmax(hL_delay)
    delayR = np.argmax(hR_delay)

    if delayL > 0:
        h_left = np.concatenate([np.zeros(delayL), h_left])
    if delayR > 0:
        h_right = np.concatenate([np.zeros(delayR), h_right])

    max_len = max(len(h_left), len(h_right))
    h_left = np.pad(h_left, (0, max_len - len(h_left)))
    h_right = np.pad(h_right, (0, max_len - len(h_right)))

    return h_left, h_right

'''
fs = 44100
c_air = 343
h_radius = 0.09
angles = [-90, -30, 0, 30, 90]

# --- 4-column figure ---
fig, axs = plt.subplots(len(angles), 4, figsize=(16, 10), sharex='col', sharey='row')
plt.subplots_adjust(hspace=0.5, wspace=0.3)

for i, ang in enumerate(angles):
    # --- Generate HRIRs + filter coeffs ---
    BL, AL, BR, AR = hrtiir(ang, h_radius, fs, c_air)
    hL, hR = hrir_gen(ang, h_radius, fs, c_air)
    nL = np.arange(len(hL))
    nR = np.arange(len(hR))

    # --- Left ear HRIR ---
    axs[i, 0].plot(nL, hL, color='tab:blue')
    axs[i, 0].set_title(f"Left Ear HRIR ({ang}°)")
    axs[i, 0].grid(True)
    axs[i, 0].set_ylabel("Amplitude")

    # --- Left ear magnitude (HRTF) ---
    w, H_L = freqz(BL, AL, fs=fs)
    axs[i, 1].semilogx(w, 20*np.log10(np.abs(H_L)), color='orange')
    axs[i, 1].set_title(f"Left ear - magnitude ({ang}°)")
    axs[i, 1].grid(True, which='both')

    # --- Right ear HRIR ---
    axs[i, 2].plot(nR, hR, color='tab:blue')
    axs[i, 2].set_title(f"Right Ear HRIR ({ang}°)")
    axs[i, 2].grid(True)

    # --- Right ear magnitude (HRTF) ---
    w, H_R = freqz(BR, AR, fs=fs)
    axs[i, 3].semilogx(w, 20*np.log10(np.abs(H_R)), color='orange')
    axs[i, 3].set_title(f"Right ear - magnitude ({ang}°)")
    axs[i, 3].grid(True, which='both')

# Labels
for j in [0, 2]:
    axs[-1, j].set_xlabel("Samples")
for j in [1, 3]:
    axs[-1, j].set_xlabel("Frequency [Hz]")
for j in [1, 3]:
    axs[0, j].set_ylim(-20, 10)

plt.suptitle("Task 4: Complete HRIR simulator", fontsize=15, y=0.99)
plt.tight_layout()
plt.show()
'''