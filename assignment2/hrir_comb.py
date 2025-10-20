import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz
from hrir import hrir          # Task 1 function (ITD only)
from hrtfiir import hrtiir     # Task 3 function (IIR filter)

def hrir_gen(inc_angle, h_radius, fs, c_air):
    BL, AL, BR, AR = hrtiir(inc_angle, h_radius, fs, c_air)

    impulse = np.zeros(512)
    impulse[0] = 1.0
    h_left_iir = lfilter(BL, AL, impulse)
    h_right_iir = lfilter(BR, AR, impulse)

    # itd delay
    hL_itd, hR_itd = hrir(inc_angle, h_radius, fs, c_air)
    delayL = np.argmax(hL_itd)
    delayR = np.argmax(hR_itd)

    # zero padding
    if delayL > 0:
        h_left = np.concatenate([np.zeros(delayL), h_left_iir])
    else:
        h_left = h_left_iir
    if delayR > 0:
        h_right = np.concatenate([np.zeros(delayR), h_right_iir])
    else:
        h_right = h_right_iir

    max_len = max(len(h_left), len(h_right))
    h_left = np.pad(h_left, (0, max_len - len(h_left)))
    h_right = np.pad(h_right, (0, max_len - len(h_right)))

    return hL_itd, hR_itd, h_left_iir, h_right_iir, h_left, h_right


fs = 44100
c_air = 343
h_radius = 0.09
angles = [-90, -45, 0, 45, 90]
sample_limit = 40

# --- Plot: Combined HRIR only ---
fig, axs = plt.subplots(len(angles), 1, figsize=(8, 10))
plt.subplots_adjust(hspace=0.5)

for i, ang in enumerate(angles):

    hL_itd, hR_itd, hL_iir, hR_iir, hL_full, hR_full = hrir_gen(ang, h_radius, fs, c_air)

    n_full = np.arange(len(hL_full))

    axs[i].plot(n_full, hL_full, color='tab:blue', label='Left')
    axs[i].plot(n_full, hR_full, color='tab:orange', linestyle='--', label='Right')
    axs[i].set_title(f"Combined HRIR ({ang}°)")
    axs[i].set_xlabel("Samples")
    axs[i].set_ylabel("Amplitude")
    axs[i].set_xlim(0, sample_limit)
    axs[i].grid(True)
    axs[i].legend()

plt.suptitle("Task 4: Combined HRIR (IIR + ITD)", fontsize=15, y=0.99)
plt.tight_layout()
plt.show()
