import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def hrir1(incidence_angle, head_radius, fs, c_air):
    angle_rad = np.deg2rad(incidence_angle)
    
    # max delay from woodworth's formula: (r/c) * π
    max_delay = head_radius * np.pi / c_air
    ir_length = int(np.ceil(max_delay * fs)) + 2
    
    irl = np.zeros(ir_length)
    irr = np.zeros(ir_length)
    
    if 0 <= incidence_angle <= 180:
        right_delay = 0
        # woodworth's formula: (r/c) * (θ + sin θ)
        left_delay = (head_radius/c_air) * (angle_rad + np.sin(angle_rad))
    else:
        left_delay = 0
        right_delay = (head_radius/c_air) * (np.pi - angle_rad + np.sin(angle_rad))
    
    right_delay_samples = int(np.round(right_delay * fs))
    left_delay_samples = int(np.round(left_delay * fs))
    
    irl[left_delay_samples] = 1
    irr[right_delay_samples] = 1
    
    return irl, irr

def hrtf1(incidence_angle, head_radius, fs, c_air, nfft):
    fvec = (fs/nfft) * np.arange(nfft//2)
    angle_rad = np.deg2rad(incidence_angle)
    
    if 0 <= incidence_angle <= 180:
        angle_to_right = np.abs(angle_rad)
        angle_to_left = np.pi - angle_to_right
    else:
        angle_to_left = np.abs(2*np.pi - angle_rad)
        angle_to_right = np.pi - angle_to_left
    
    # alpha varies from 0.1 (shadow) to 1.0 (direct)
    alpha_right = 0.1 + 0.9 * (1 - angle_to_right/np.pi)
    alpha_left = 0.1 + 0.9 * (1 - angle_to_left/np.pi)
    
    beta = c_air / head_radius  # corner frequency
    
    omega = 2 * np.pi * fvec
    jw = 1j * omega
    
    # brown-duda shelving filter
    tfl = (alpha_left * jw + beta) / (jw + beta)
    tfr = (alpha_right * jw + beta) / (jw + beta)
    
    return tfl, tfr, fvec

def hrtfiir(incidence_angle, head_radius, fs, c_air):
    angle_rad = np.deg2rad(incidence_angle)
    
    if 0 <= incidence_angle <= 180:
        angle_to_right = np.abs(angle_rad)
        angle_to_left = np.pi - angle_to_right
    else:
        angle_to_left = np.abs(2*np.pi - angle_rad)
        angle_to_right = np.pi - angle_to_left
    
    alpha_right = 0.1 + 0.9 * (1 - angle_to_right/np.pi)
    alpha_left = 0.1 + 0.9 * (1 - angle_to_left/np.pi)
    
    beta = c_air / head_radius
    beta_prewarped = 2 * fs * np.tan(beta / (2 * fs))  # bilinear transform prewarping
    
    BL = np.array([
        (alpha_left * beta_prewarped + beta) / (beta_prewarped + beta),
        (alpha_left * beta_prewarped - beta) / (beta_prewarped + beta)
    ])
    AL = np.array([1.0, (beta_prewarped - beta) / (beta_prewarped + beta)])
    
    BR = np.array([
        (alpha_right * beta_prewarped + beta) / (beta_prewarped + beta),
        (alpha_right * beta_prewarped - beta) / (beta_prewarped + beta)
    ])
    AR = np.array([1.0, (beta_prewarped - beta) / (beta_prewarped + beta)])
    
    return BL, AL, BR, AR

def hrir_complete(x, incidence_angle, head_radius, fs, c_air):
    irl, irr = hrir1(incidence_angle, head_radius, fs, c_air)
    BL, AL, BR, AR = hrtfiir(incidence_angle, head_radius, fs, c_air)
    
    x_padded = np.pad(x, (0, len(irl)-1), mode='constant')
    
    # apply itd through convolution
    y_left_delay = np.convolve(x_padded, irl)
    y_right_delay = np.convolve(x_padded, irr)
    
    # apply shelving filter
    y_left = signal.lfilter(BL, AL, y_left_delay)
    y_right = signal.lfilter(BR, AR, y_right_delay)
    
    return y_left[:len(x)], y_right[:len(x)]
    

def generate_pink_noise(duration, fs):
    num_samples = int(duration * fs)
    white = np.random.normal(0, 1, num_samples)
    
    freqs = np.fft.rfftfreq(num_samples)
    freqs[0] = 1e-6  # avoid divide by zero
    
    # apply 1/sqrt(f) filter to create pink noise
    pink_f = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(pink_f)
    
    return pink / np.max(np.abs(pink))

def demo_spatial_audio():
    fs = 44100
    head_radius = 0.09
    c_air = 343
    
    # impulse responses
    angles = [0, 45, 90, 135, 180]
    plt.figure(figsize=(12, 4))
    for angle in angles:
        irl, irr = hrir1(angle, head_radius, fs, c_air)
        plt.plot(irl, label=f'Left {angle}°')
        plt.plot(irr, label=f'Right {angle}°')
    plt.title('Impulse Responses')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # frequency responses
    plt.figure(figsize=(12, 4))
    nfft = 1024
    for angle in angles:
        tfl, tfr, fvec = hrtf1(angle, head_radius, fs, c_air, nfft)
        plt.semilogx(fvec, 20*np.log10(np.abs(tfl)), label=f'Left {angle}°')
        plt.semilogx(fvec, 20*np.log10(np.abs(tfr)), label=f'Right {angle}°')
    plt.title('Frequency Responses')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # moving noise bursts
    burst_duration = 0.2
    gap_duration = 0.1
    angles = np.arange(0, 360, 30)
    
    burst_samples = int(burst_duration * fs)
    gap_samples = int(gap_duration * fs)
    total_samples = (burst_samples + gap_samples) * len(angles)
    y_left_total = np.zeros(total_samples)
    y_right_total = np.zeros(total_samples)
    
    for i, angle in enumerate(angles):
        burst = generate_pink_noise(burst_duration, fs)
        y_left, y_right = hrir_complete(burst, angle, head_radius, fs, c_air)
        
        start_idx = i * (burst_samples + gap_samples)
        y_left_total[start_idx:start_idx + burst_samples] = y_left
        y_right_total[start_idx:start_idx + burst_samples] = y_right
    
    plt.figure(figsize=(12, 4))
    t = np.arange(len(y_left_total)) / fs
    plt.plot(t, y_left_total, label='Left')
    plt.plot(t, y_right_total, label='Right')
    plt.title('Moving Pink Noise Bursts')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    return y_left_total, y_right_total, fs

if __name__ == "__main__":
    try:
        import sounddevice as sd
        has_audio = True
    except ImportError:
        print("Note: sounddevice module not found. Audio playback will be disabled.")
        has_audio = False
    
    # Run the complete demonstration
    y_left, y_right, fs = demo_spatial_audio()
    
    # Show all plots
    plt.show()
    
    # Play the spatial audio if sounddevice is available
    if has_audio:
        print("\nPlaying spatial audio demonstration...")
        print("You should hear pink noise bursts moving around your head in 30-degree steps.")
        print("Please wear headphones for the best spatial effect.")
        stereo_output = np.vstack((y_left, y_right)).T
        sd.play(stereo_output, fs)
        sd.wait()
        print("Playback complete!")