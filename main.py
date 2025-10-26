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
    fft_vals = np.abs(np.fft.fft(audio))
    # 1/sr is the time between samples
    freq_axis = np.fft.fftfreq(len(audio), 1/sr)

    plt.subplot(2, 1, 2)
    # Plot frequency plot
    plt.plot(freq_axis, fft_vals)
    plt.title(f"Waveform: {filename}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, max(freq_axis)) # Only plot the positive half of the FFT

    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Please provide a .mp3 file as the first argument")
        return

    filename = sys.argv[1]

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

    plot_song(audio, filename, sr)

if __name__ == "__main__":
    main()
