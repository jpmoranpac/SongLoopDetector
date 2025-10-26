import sys
import librosa
import matplotlib.pyplot as plt

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

    # Create time axis for plotting
    time_axis = librosa.times_like(audio, sr=sr)

    # Plot waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, audio)
    plt.title(f"Waveform: {filename}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
