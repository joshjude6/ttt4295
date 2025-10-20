import numpy as np
import matplotlib.pyplot as plt

def hrtf1(inc_angle, h_radius, fs, c_air, nfft):
    theta = np.deg2rad(inc_angle)
    fvec = np.linspace(0, fs/2, nfft//2 + 1)
    beta = (2*c_air) / h_radius
    omega = 2 * np.pi * fvec
    
    alpha_left = 1 - np.sin(theta)
    alpha_right = 1 + np.sin(theta)

    tfl = (alpha_left * 1j * omega + beta) / (1j * omega + beta)
    tfr = (alpha_right * 1j * omega + beta) / (1j * omega + beta)
    
    return tfl, tfr, fvec

'''
fs = 44100
c_air = 343
h_radius = 0.09
nfft = 2048
angles = [-90, 0, 90]


fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

for ang in angles:
    H_L, H_R, fvec = hrtf1(ang, h_radius, fs, c_air, nfft)
    
    axs[0].semilogx(fvec, 20*np.log10(np.abs(H_L)), label=f'{ang}°')
    axs[1].semilogx(fvec, 20*np.log10(np.abs(H_R)), label=f'{ang}°', linestyle='--')

axs[0].set_title('Left ear HRTFs')
axs[1].set_title('Right ear HRTFs')

axs[1].set_xlabel('Frequency [Hz]')
for ax in axs:
    ax.set_ylabel('Magnitude [dB]')
    ax.grid(True, which='both')
    ax.legend(title='Incidence angle')

plt.suptitle('Task 2: Brown & Duda HRTFs', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
'''