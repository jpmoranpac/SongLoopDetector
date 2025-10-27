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

def main():
    if len(sys.argv) != 2:
        print("Please provide a .mp3 file as the first argument")
        return

    filename = sys.argv[1]

    audio, sr = load_audio(filename)

    fft_matching(audio, sr, 0.99, 1.0, 0.5, 0.0)


if __name__ == "__main__":
    main()
