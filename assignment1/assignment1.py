## Code for Assignment 1 in TTT4295, autumn 2025
## Written by Josh Jude

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.io import wavfile
import os
import glob

def analyze_harmonics(filename, threshold=0.1, n_fft=16384, fmin=20.0, plot=False):
    rate, data = wavfile.read(filename)
    data = data[:, 0]
    n = len(data)

    if n < n_fft:
        x = np.pad(data, (0, n_fft - n))
    else:
        x = data[:n_fft]

    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(n_fft, 1/rate)
    mag = np.abs(X)
    freqs = freqs[:n_fft // 2]
    mag = mag[:n_fft // 2]

    peaks, _ = find_peaks(mag) # finding the local maxima
    max_mag = mag[peaks].max() # strongest peak magnitude
    strong = [p for p in peaks if mag[p] >= threshold * max_mag] # keeping peaks above a certain threshold
    strong_sorted = sorted(strong, key=lambda i: freqs[i]) # sorting by frequency
    strong_sorted = [i for i in strong_sorted if freqs[i] >= fmin] # discarding weaker peaks

    f0 = freqs[strong_sorted[0]] # fundamental frequency is the lowest freq peak
    mag0 = mag[strong_sorted[0]] # magnitude of fundamental peak

    tolerance_hz = rate/n_fft # the bin wdith

    groupA, groupB = [], [] # splitting into harmonics and non-harmonics
    for idx in strong_sorted: # checking every frequency that was kept
        f = freqs[idx] # peak freq
        if f < f0 - tolerance_hz: # skipping if the freq is below fundamental
            continue
        k = int(round(f / f0)) # finding the harmonic number
        if k < 1: # adding to group B if less than fundamental
            groupB.append({"frequency": f, "magnitude": mag[idx]})
            continue

        f_ideal = k * f0 # calculating the deviation from an ideal harmonic frequency
        dev_hz = f - f_ideal
        if abs(dev_hz) <= tolerance_hz: # if it is close enough, calculate the deviations and add to group A

            dev_cents = 1200.0 * np.log2(f/f_ideal)
            level_db = 20.0 * np.log10(mag[idx]/mag0)
            groupA.append({
                "k": k,
                "frequency": f,
                "magnitude": mag[idx],
                "deviation_hz": dev_hz,
                "deviation_cents": dev_cents,
                "level_db_rel_f0": level_db
            })

        else:
            groupB.append({"frequency": f, "magnitude": mag[idx]}) # if not close enough, add to group B

    return {
        "f0": f0,
        "tolerance_hz": tolerance_hz,
        "groupA": sorted(groupA, key=lambda d: d["k"]),
        "groupB": groupB
    }

def frequency_to_note_and_cents(frequency):
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    A4 = 440.0
    semitones_from_A4 = 12 * np.log2(frequency / A4)
    closest_semitone = round(semitones_from_A4)
    
    note_index = (9 + closest_semitone) % 12
    octave = 4 + (9 + closest_semitone) // 12
    note_name = note_names[note_index]
    
    theoretical_freq = A4 * (2 ** (closest_semitone / 12))
    cents_deviation = 1200 * np.log2(frequency / theoretical_freq)
    return f"{note_name}{octave}", theoretical_freq, cents_deviation

def process_multiple_files(files, output_file="harmonic_analysis_results.txt", **kwargs):

    files = glob.glob(files)
    files.sort()
    
    if not files:
        print(f"No files found matching pattern: {files}")
        return
    
    print(f"Found {len(files)} files to process")
    
    with open(output_file, 'w') as f:
        f.write("Harmonic analysis\n")
        f.write("=" * 50 + "\n\n")
        
        for i, filename in enumerate(files, 1):
            print(f"Processing {i}/{len(files)}: {os.path.basename(filename)}")
            
            try:
                # Analyze the file
                result = analyze_harmonics(filename, **kwargs)
                
                # Write results to file
                f.write(f"File {i}: {os.path.basename(filename)}\n")
                f.write("-" * 40 + "\n")
                
                if result["f0"] is None:
                    f.write("No fundamental frequency detected.\n\n")
                    continue
                
                # Get musical note information
                note, theoretical_freq, cents_dev = frequency_to_note_and_cents(result["f0"])
                
                f.write(f"Fundamental frequency (f0): {result['f0']:.3f} Hz\n")
                f.write(f"Frequency resolution: {result['tolerance_hz']:.3f} Hz\n")
                f.write(f"Tolerance: ±{result['tolerance_hz']:.3f} Hz\n")
                
                if note:
                    f.write(f"Musical note: {note}\n")
                    f.write(f"Theoretical frequency: {theoretical_freq:.3f} Hz\n")
                    f.write(f"Deviation from equal temperament: {cents_dev:+.2f} cents\n")
                
                f.write(f"\nGroup A (Harmonics): {len(result['groupA'])} peaks\n")
                f.write("k   Frequency [Hz]   Dev [Hz]   Dev [cents]   Level [dB]\n")
                for p in result["groupA"]:
                    f.write(f"{p['k']:>2d}  {p['frequency']:>11.3f}   {p['deviation_hz']:>+8.3f}   {p['deviation_cents']:>+8.2f}   {p['level_db_rel_f0']:>+8.2f}\n")
                
                if result["groupB"]:
                    f.write(f"\nGroup B (Non-harmonic): {len(result['groupB'])} peaks\n")
                    f.write("Frequencies [Hz]: ")
                    f.write(", ".join(f"{p['frequency']:.2f}" for p in result["groupB"]))
                    f.write("\n")
                
                f.write("\n" + "="*50 + "\n\n")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                f.write(f"ERROR processing file: {e}\n\n")
    
    print(f"Results written to: {output_file}")

def create_summary_table(files, output_file="summary_table.txt", **kwargs):
    files = glob.glob(files)
    files.sort()
    
    if not files:
        print(f"No files found matching pattern: {files}")
        return
    
    with open(output_file, 'w') as f:
        f.write("Fundamental frequencies and musical notes\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'File':<30} {'f0 [Hz]':<10} {'Note':<6} {'Theoretical [Hz]':<15} {'Deviation [cents]':<15}\n")
        f.write("-" * 80 + "\n")
        
        for filename in files:
            try:
                result = analyze_harmonics(filename, plot=False, **kwargs)
                
                if result["f0"] is None:
                    f.write(f"{os.path.basename(filename):<30} {'N/A':<10} {'N/A':<6} {'N/A':<15} {'N/A':<15}\n")
                    continue
                
                note, theoretical_freq, cents_dev = frequency_to_note_and_cents(result["f0"])
                
                f.write(f"{os.path.basename(filename):<30} {result['f0']:<10.3f} {note or 'N/A':<6} ")
                f.write(f"{theoretical_freq or 0:<15.3f} {cents_dev or 0:<+15.2f}\n")
                
            except Exception as e:
                f.write(f"{os.path.basename(filename):<30} ERROR: {str(e)}\n")
    
    print(f"Summary table written to: {output_file}")


if __name__ == "__main__":
    files = "/Users/joshjude/Documents/Git/ttt4295/assignment1/music_box_tones/*.wav"
    process_multiple_files(
        files,
        output_file="/assignment1/detailed_harmonic_analysis.txt",
        threshold=0.2,
        n_fft=65536,
        plot=False
    )
    
    create_summary_table(
        files,
        output_file="/assignment1/summary_table.txt",
        threshold=0.2,
        n_fft=65536
    )
