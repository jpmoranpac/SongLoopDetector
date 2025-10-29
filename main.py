import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np

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

def main():
    if len(sys.argv) != 2:
        print("Please provide a .mp3 file as the first argument")
        return

    filename = sys.argv[1]

    audio, sr = load_audio(filename)

    #fft_matching(audio, sr, 0.99, 1.0, 0.5, 0.0)


    # Run sliding cross-correlation
    lags, scores = sliding_cross_correlation(audio, sr, offset=5.0, window_duration=5.0, hop_size=1)

    # Find best loop point
    best_idx = np.argmax(scores)
    best_time = lags[best_idx]
    best_score = scores[best_idx]

    print(f"Best loop point at: {best_time:.2f} seconds (similarity = {best_score:.4f})")

    # Plot the correlation scores
    plt.figure(figsize=(10, 4))
    plt.plot(lags, scores)
    plt.title("Cross-Correlation vs Time Offset")
    plt.xlabel("Time Offset (seconds)")
    plt.ylabel("Similarity Score")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
