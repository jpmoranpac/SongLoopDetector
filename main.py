# Steps:
# [x] Use FFT to find best matching loop for a given section
# [x] Slide the window out to find the longest period of matching samples (with
#     a criteria of 0.98 match?)
# [x] Repeat this for all sections of x seconds in the song
# [x] Plot the sections of songs that appear to be repeats
# [ ] Export a new .mp3 file that stitches the matching sections together
# [ ] Interactive GUI for the user to confirm repeats (e.g. click on a proposed
#     repeat and let the user hear the proposed stitch for seamlessness)

# Issues:
# [x] As the window duration increases, the average correlation increases
#     Makes sense, as the whole song is similar to itself. Should I
#     instead look for number of sequential matching short windows?
# [ ] How to handle songs with multiple valid loop points?

import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from pathlib import Path

def plot_song(audio, filename, sr):
    # Create time axis for plotting
    time_axis = librosa.times_like(audio, sr=sr)

    plt.figure(figsize=(12, 4))

    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, audio)
    plt.title(f"Waveform: {filename}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.xlim(0, max(time_axis))

    # Create frequency axis for plotting
    fft_vals = np.abs(np.fft.rfft(audio))
    # 1/sr is the time between samples
    freq_axis = np.fft.rfftfreq(len(audio), 1/sr)

    plt.subplot(2, 1, 2)
    # Plot frequency plot
    plt.plot(freq_axis, fft_vals)
    plt.title(f"Waveform: {filename}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, max(freq_axis)) # Only plot the positive half of the FFT

    plt.tight_layout()
    plt.show()

def plot_song_with_matches(audio, filename, sr, matching_samples, step=1):
    # Downsample for faster plotting
    audio_ds = audio[::step]
    matching_ds = matching_samples[::step]
    time_ds = librosa.times_like(audio_ds, sr=sr, hop_length=step)

    # Find unique region IDs and assign colours
    unique_ids = np.unique(matching_ds)
    cmap = plt.get_cmap("tab20", len(unique_ids))
    color_map = {uid: cmap(i) for i, uid in enumerate(unique_ids)}

    plt.figure(figsize=(12, 4))

    # Draw contiguous segments in the same colour
    start = 0
    for i in range(1, len(audio_ds)):
        if matching_ds[i] != matching_ds[start] or i == len(audio_ds) - 1:
            plt.plot(time_ds[start:i], audio_ds[start:i],
                     color=color_map[matching_ds[start]], linewidth=0.8)
            start = i

    plt.title(f"Waveform with Match Regions: {filename}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.xlim(0, max(time_ds))
    plt.tight_layout()
    plt.show()

def compare_sections(audio1, audio2, filename, sr):
    # Create time axis for plotting
    time_axis = librosa.times_like(audio1, sr=sr)

    plt.figure(figsize=(12, 4))

    # Plot waveform
    plt.subplot(2, 2, 1)
    plt.plot(time_axis, audio1)
    plt.title(f"Waveform: {filename}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.xlim(0, max(time_axis))

    # Create frequency axis for plotting
    fft_vals = np.abs(np.fft.rfft(audio1))
    # 1/sr is the time between samples
    freq_axis = np.fft.rfftfreq(len(audio1), 1/sr)

    plt.subplot(2, 2, 3)
    # Plot frequency plot
    plt.plot(freq_axis, fft_vals)
    plt.title(f"Waveform: {filename}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, max(freq_axis)) # Only plot the positive half of the FFT

    # Plot waveform
    plt.subplot(2, 2, 2)
    plt.plot(time_axis, audio2)
    plt.title(f"Waveform: {filename}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.xlim(0, max(time_axis))

    # Create frequency axis for plotting
    fft_vals = np.abs(np.fft.rfft(audio2))
    # 1/sr is the time between samples
    freq_axis = np.fft.rfftfreq(len(audio2), 1/sr)

    plt.subplot(2, 2, 4)
    # Plot frequency plot
    plt.plot(freq_axis, fft_vals)
    plt.title(f"Waveform: {filename}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, max(freq_axis)) # Only plot the positive half of the FFT

    plt.tight_layout()
    plt.show()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def load_audio(filename):
    print(f"Loading audio file: {filename}")
    try:
        # Load the audio file in mono, preserving the original sample rate
        audio, sr = librosa.load(filename, sr=None, mono=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    duration = len(audio) / sr
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration:.2f} seconds")

    return audio, sr

def fft_matching(audio, sr, similarity, window_sec, hop_sec, percentage_skip):
    duration = len(audio) / sr
    window_size = int(window_sec * sr)
    hop_size = int(hop_sec * sr)
    windows = []
    for start in range(0, len(audio) - window_size, hop_size):
        segment = audio[start:start + window_size]
        fft = np.abs(np.fft.rfft(segment))
        windows.append((start, fft))
    
    for i1, w1 in enumerate(windows):
        for w2 in windows[i1 + int(len(audio)*percentage_skip):]:
            if cosine_similarity(w1[1], w2[1]) > similarity and w1[0] != w2[0]:
                print(f"Match {w1[0]/sr} with {w2[0]/sr}!")

def sliding_cross_correlation(audio, sr, offset=15.0, window_duration=2.0, hop_size=1):
    # This method often returns quite a poor correlation, even for loops that
    # are obvious to the human ear. Really good for mechanical loops, but 
    # natural ones get correlation scores below 0.5. Could this be beacuse of 
    # phase shifts?

    window_size = int(window_duration * sr)
    offset_samples = int(offset * sr)
    ref = audio[offset_samples:offset_samples+window_size]  # reference window

    correlation_scores = []
    lag_times = []
    segments_to_check = int((len(audio) - window_size)/hop_size)
    print(f"Checking {segments_to_check} segments")

    for lag in range(window_size + offset_samples, len(audio) - window_size, hop_size):
        segment = audio[lag:lag + window_size]

        # Compute normalized dot product (cosine similarity)
        score = cosine_similarity(ref, segment)
        correlation_scores.append(score)
        lag_times.append(lag / sr)  # store in seconds

        if lag % 10000 == 0:
            print(f"{(lag)/len(audio)*100:.2f}%")

    return np.array(lag_times), np.array(correlation_scores)

def calculate_similarity(ref, segment):
    ref_mag = np.abs(librosa.stft(ref, n_fft=2048, hop_length=512)).mean(axis=1)
    ref_mag /= np.linalg.norm(ref_mag)
    seg_mag = np.abs(librosa.stft(segment, n_fft=2048, hop_length=512)).mean(axis=1)
    seg_mag /= np.linalg.norm(seg_mag) + 1e-8

    score = np.dot(ref_mag, seg_mag)

    return score

def frequency_cross_correlation(audio, reference_sample, search_offset_sample, window_size, hop_size=1):
    ref = audio[reference_sample:reference_sample+window_size]  # reference window
    
    ref_mag = np.abs(librosa.stft(ref, n_fft=2048, hop_length=512)).mean(axis=1)
    ref_mag /= np.linalg.norm(ref_mag + 1e-8)
    
    scores = []
    lags = []

    for lag in range(search_offset_sample, len(audio) - window_size, hop_size):
        segment = audio[lag:lag + window_size]
        seg_mag = np.abs(librosa.stft(segment, n_fft=2048, hop_length=512)).mean(axis=1)
        seg_mag /= np.linalg.norm(seg_mag) + 1e-8

        score = np.dot(ref_mag, seg_mag)
        scores.append(score)
        lags.append(lag)
            
    return np.array(lags), np.array(scores)

def plot_similarity_scores(lags, scores):
    # Plot the correlation scores
    plt.figure(figsize=(10, 4))
    plt.plot(lags, scores)
    plt.title("Cross-Correlation vs Time Offset")
    plt.xlabel("Time Offset (seconds)")
    plt.ylabel("Similarity Score")
    plt.grid(True)
    plt.show()

def find_consecutive_matching_samples(audio, reference_start_sample, match_start_sample, window_size, similarity_threshold, allowable_misses = 0):
    similarity = 1.0
    num_consecutive_matches = 0
    misses = 0
    while misses <= allowable_misses:
        # Check if the next window is also of high similarity
        num_consecutive_matches += 1
        ref_start = reference_start_sample + window_size * num_consecutive_matches
        ref_end = ref_start + window_size
        sample_start = match_start_sample + window_size * num_consecutive_matches
        sample_end = sample_start + window_size
        ref = audio[ref_start : ref_end]
        sample = audio[sample_start : sample_end]
        similarity = calculate_similarity(ref, sample)
        if similarity < similarity_threshold:
            misses += 1
        else:
            misses = 0

    reference_end_sample = reference_start_sample + num_consecutive_matches * window_size
    match_end_sample = match_start_sample + num_consecutive_matches * window_size
    suitable_loop_cut = False
    if (reference_end_sample >= match_start_sample):
        suitable_loop_cut = True
        
    return num_consecutive_matches, suitable_loop_cut, reference_end_sample, match_end_sample

def find_first_loop_point(audio, sr, window_duration, similarity_threshold):
    found_suitable_loop = False
    current_offset = int(0.0 * sr)

    # Find points where similarity is high
    while (found_suitable_loop == False and current_offset < len(audio)):
        print(f"Checking {current_offset / sr}")
        lags, scores = frequency_cross_correlation(audio, current_offset, current_offset + window_duration, window_duration, hop_size=10000)
        matching_sample = [0] * len(audio)
        for idx, score in enumerate(scores):
            if score > similarity_threshold:
                consecutive_matches, found_suitable_loop, reference_end_sample, match_end_sample = find_consecutive_matching_samples(audio, current_offset, lags[idx], window_duration, similarity_threshold)
                print(f"For reference at {current_offset:.2f} to {reference_end_sample:.2f} Found {consecutive_matches} consecutive matches, starting at {lags[idx]:.2f} to {match_end_sample:.2f}, suitable loop? {found_suitable_loop}")
            if found_suitable_loop:
                matching_sample[current_offset:reference_end_sample] = [-current_offset] * consecutive_matches * window_duration
                matching_sample[lags[idx]:match_end_sample] = [idx] * consecutive_matches * window_duration
                break
        if not found_suitable_loop:
            current_offset += window_duration

    return found_suitable_loop, matching_sample, current_offset, reference_end_sample, lags[idx], match_end_sample

def main():
    # Load file
    if len(sys.argv) != 2:
        print("Please provide a .mp3 file as the first argument")
        return
    filename = sys.argv[1]
    audio, sr = load_audio(filename)

    # Analysis settings
    window_duration = int(0.5 * sr)
    similarity_threshold = 0.99

    found_suitable_loop, matching_sample, reference_start, reference_end, match_start, match_end = find_first_loop_point(audio, sr, window_duration, similarity_threshold)

    if (found_suitable_loop):
        plot_song_with_matches(audio, filename, sr, matching_sample, 1_000)
    else:
        print("No suitable loop found")

    # Find the exact loop point
    print("Finding exact loop point...")
    # lags, scores = frequency_cross_correlation(audio[:match_start+window_duration * 2], reference_start, match_start, window_duration, hop_size=10)
    # best_loop = lags[np.argmax(scores)]
    # print(f"Moved loop from {match_start} to {best_loop}")
    best_loop = match_start

    # Remove the identified loop from the song
    print(f"Stiching {reference_start} to {best_loop}")
    audio_loop_removed = np.concatenate([audio[:reference_start], audio[best_loop:]])

    print(f"Stitched length: {len(audio_loop_removed)/sr:.2f}s")

    # Write out as .wav
    output_path = "loop_removed_" + Path(filename).stem + ".wav"
    sf.write(output_path, audio_loop_removed, sr, subtype="PCM_16")
    print(f"Exported stitched audio: {output_path}")

if __name__ == "__main__":
    main()
