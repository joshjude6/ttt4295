import numpy as np
import matplotlib.pyplot as plt


def hrir(inc_angle, h_radius, fs, c_air):
    theta = np.deg2rad(inc_angle)
    delta_t = (h_radius / c_air) * (theta + np.sin(theta))
    delta_samples = int(np.round(abs(delta_t) * fs))

    ir_length = delta_samples + 1
    h_left = np.zeros(ir_length)
    h_right = np.zeros(ir_length)
    
    # sunny side ear gets impulse at sample 0
    if theta >= 0:
        # sound comes from the right
        h_right[0] = 1.0
        h_left[delta_samples] = 1.0  
    elif theta < 0:
        # sound comes from the left
        h_left[0] = 1.0
        h_right[delta_samples] = 1.0
    else:
        h_left[0] = 1.0
        h_right[0] = 1.0
    
    return h_left, h_right

'''
# Parameters
fs = 44100
c_air = 343
h_radius = 0.09
angles = [-90, 0, 90]

# Create subplot grid: one row per angle, 2 columns (left/right)
fig, axs = plt.subplots(len(angles), 2, figsize=(10, 6), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.4)

for i, ang in enumerate(angles):
    hL, hR = hrir(ang, h_radius, fs, c_air)
    tL = np.arange(len(hL)) / fs * 1000
    tR = np.arange(len(hR)) / fs * 1000

    axs[i, 0].stem(tL, hL, linefmt='tab:blue', markerfmt='bo', basefmt=" ")
    axs[i, 1].stem(tR, hR, linefmt='tab:red', markerfmt='ro', basefmt=" ")
    
    axs[i, 0].set_title(f'Left ear (Incoming angle: {ang}°)')
    axs[i, 1].set_title(f'Right ear (Incoming angle: {ang}°)')
    axs[i, 0].set_ylabel('Amplitude')

for ax in axs[-1, :]:
    ax.set_xlabel('Time [ms]')

for ax in axs.flat:
    ax.grid(True)

plt.suptitle("Task 1: Head-related impulse responses", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()
'''