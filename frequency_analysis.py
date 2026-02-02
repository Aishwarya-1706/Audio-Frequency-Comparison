import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load two audio files
audio1, sr1 = librosa.load("audio1.wav", sr=None)
audio2, sr2 = librosa.load("audio2.wav", sr=None)

# Apply FFT
fft_audio1 = np.abs(np.fft.fft(audio1))
fft_audio2 = np.abs(np.fft.fft(audio2))

# Frequency axis
freq1 = np.fft.fftfreq(len(fft_audio1), 1/sr1)
freq2 = np.fft.fftfreq(len(fft_audio2), 1/sr2)

# Plot frequency spectrum
plt.figure(figsize=(10, 5))
plt.plot(freq1[:len(freq1)//2], fft_audio1[:len(fft_audio1)//2], label="Audio 1")
plt.plot(freq2[:len(freq2)//2], fft_audio2[:len(fft_audio2)//2], label="Audio 2")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Domain Comparison")
plt.legend()
plt.show()


from numpy.linalg import norm

# Make both FFTs same length
min_len = min(len(fft_audio1), len(fft_audio2))
fft1 = fft_audio1[:min_len]
fft2 = fft_audio2[:min_len]

# Cosine similarity
similarity = np.dot(fft1, fft2) / (norm(fft1) * norm(fft2))

print("Similarity Score:", similarity)

from numpy.linalg import norm

# Make both FFTs same length
min_len = min(len(fft_audio1), len(fft_audio2))
fft1 = fft_audio1[:min_len]
fft2 = fft_audio2[:min_len]

# Cosine similarity
similarity = np.dot(fft1, fft2) / (norm(fft1) * norm(fft2))

print("Similarity Score:", similarity)
