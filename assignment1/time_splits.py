import numpy as np
from scipy.io import wavfile
import os

def split_audio_file(input_filename, time_splits, output_dir="split_audio"):
    """
    Split an audio file into multiple segments based on time intervals.
    
    Parameters:
    input_filename (str): Path to the input audio file
    time_splits (list): List of [start_time, end_time] pairs in seconds
    output_dir (str): Directory to save the split files
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the audio file
    rate, data = wavfile.read(input_filename)
    
    # Handle stereo files by taking first channel
    if data.ndim > 1:
        data = data[:, 0]
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    
    print(f"Original file: {input_filename}")
    print(f"Sample rate: {rate} Hz")
    print(f"Duration: {len(data)/rate:.2f} seconds")
    print(f"Splitting into {len(time_splits)} segments...")
    
    for i, (start_time, end_time) in enumerate(time_splits):
        # Convert time to sample indices
        start_sample = int(start_time * rate)
        end_sample = int(end_time * rate)
        
        # Bounds checking
        start_sample = max(0, start_sample)
        end_sample = min(len(data), end_sample)
        
        # Extract the segment
        segment = data[start_sample:end_sample]
        
        # Create output filename with zero-padded numbers
        output_filename = os.path.join(output_dir, f"{base_name}_{i+1:03d}.wav")
        
        # Save the segment
        wavfile.write(output_filename, rate, segment.astype(data.dtype))
        
        duration = (end_sample - start_sample) / rate
        print(f"Segment {i+1:2d}: {start_time:6.2f}s - {end_time:6.2f}s ({duration:5.2f}s) -> {os.path.basename(output_filename)}")
    
    print(f"\nAll segments saved to: {output_dir}/")

# Your time splits
time_splits = [
    [14.83, 18.35],
    [18.35, 21.27],
    [21.27, 33.97],
    [33.97, 41.64],
    [41.64, 48.09],
    [48.09, 56.97],
    [56.97, 64.95],
    [64.95, 68.78],
    [68.78, 71.27],
    [71.27, 76.85],
    [76.85, 81.33],
    [81.33, 87.61],
    [87.61, 90.95],
    [90.95, 91.83],
    [91.83, 93.98],
    [93.98, 98.04],
    [98.04, 101.92],
    [101.92, 108.03],
    [108.03, 111.65],
    [111.65, 113.31],
    [113.31, 116.96],
    [116.96, 117.00],
    [117.00, 127.09],
    [127.09, 130.30],
    [130.30, 131.95],
    [131.95, 135.96],
    [135.96, 139.64],
    [139.64, 141.65],
    [141.65, 146.21],
    [146.21, 158.09],
    [158.09, 164.04],
    [164.04, 173.11],
    [173.11, 176.29],
    [176.29, 182.16],
    [182.16, 188.33],
    [188.33, 192.45],
    [192.45, 194.17],
    [194.17, 202.20],
    [202.20, 209.10],
    [209.10, 214.55],
    [214.55, 215.93],
    [215.93, 219.33],
    [219.33, 223.32],
    [223.32, 228.13],
    [228.13, 231.50],
    [231.50, 233.69],
    [233.69, 236.89],
    [236.89, 237.72],
    [237.72, 243.66],
    [243.66, 244.42],
    [244.42, 249.97],
    [249.97, 255.92],
    [255.92, 261.33],
    [261.33, 263.96],
    [263.96, 267.14],
    [267.14, 270.32],
    [270.32, 273.00],
    [273.00, 276.00],
]

# Usage example:
if __name__ == "__main__":
    # Replace with your actual input file path
    input_file = "/Users/joshjude/Documents/Git/ttt4295/assignment1/pink-panther.wav"
    
    # Split the audio file
    split_audio_file(input_file, time_splits, output_dir="music_box_tones_k")