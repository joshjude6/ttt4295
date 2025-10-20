import numpy as np
import matplotlib.pyplot as plt

def hrtiir(inc_angle, h_radius, fs, c_air):
    theta = np.deg2rad(inc_angle)
    T = 1 / fs
    beta = (2 * c_air) / h_radius
    alpha_left = 1 - np.sin(theta)
    alpha_right = 1 + np.sin(theta)

    A_0 = T*beta + 2
    A_1 = (T*beta - 2)/A_0

    # left side
    B_0_left = ((T*beta) + (2*alpha_left))/A_0
    B_1_left = ((T*beta) - (2*alpha_left))/A_0

    # right side
    B_0_right = ((T*beta) + (2*alpha_right))/A_0
    B_1_right = ((T*beta) - (2*alpha_right))/A_0

    # packing into arrays
    BL = np.array([B_0_left, B_1_left])
    BR = np.array([B_0_right, B_1_right])
    AL = np.array([1, A_1])
    AR = np.array([1, A_1])

    return BL, AL, BR, AR


if __name__ == "__main__":
    from scipy.signal import freqz
    
    fs = 44100
    c_air = 343
    h_radius = 0.09
    angles = [-90, -30, 0, 30, 90]

    f_min, f_max = 20, fs/2
    f = np.logspace(np.log10(f_min), np.log10(f_max), 2048)
    w = 2 * np.pi * f / fs

    beta = 2.0 * c_air / h_radius
    f_c = beta / (2*np.pi)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    
    for ang in angles:
        BL, AL, BR, AR = hrtiir(ang, h_radius, fs, c_air)

        _, HL = freqz(BL, AL, worN=w)
        _, HR = freqz(BR, AR, worN=w)

        HLdb = 20*np.log10(np.maximum(np.abs(HL), 1e-12))
        HRdb = 20*np.log10(np.maximum(np.abs(HR), 1e-12))

        ax1.semilogx(f, HLdb, label=f"{ang}°")
        ax2.semilogx(f, HRdb, label=f"{ang}°")

    for ax in [ax1, ax2]:
        ax.axvline(f_c, color='k', linestyle=':', linewidth=1)
        ax.set_xlim([f_min, f_max])
        ax.set_ylim([-30, 8])
        ax.grid(True, which='both', linestyle=':')
        ax.legend()
    
    ax1.set_ylabel("Magnitude [dB]")
    ax1.set_title("Left ear")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Magnitude [dB]")
    ax2.set_title("Right ear")
    fig.suptitle("Task 3: Brown–Duda IIR magnitude")
    
    plt.tight_layout()
    plt.show()