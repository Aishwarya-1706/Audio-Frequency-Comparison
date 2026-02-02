import numpy as np
from scipy.io import wavfile
from numpy.linalg import norm

# Load audio files
rate1, audio1 = wavfile.read("audio1.wav")
rate2, audio2 = wavfile.read("audio2.wav")

# Convert stereo to mono if needed
if len(audio1.shape) > 1:
    audio1 = audio1.mean(axis=1)
if len(audio2.shape) > 1:
    audio2 = audio2.mean(axis=1)

# Apply FFT
fft_audio1 = np.abs(np.fft.fft(audio1))
fft_audio2 = np.abs(np.fft.fft(audio2))

# Make both FFTs same length
min_len = min(len(fft_audio1), len(fft_audio2))
fft1 = fft_audio1[:min_len]
fft2 = fft_audio2[:min_len]

# Cosine similarity
similarity = np.dot(fft1, fft2) / (norm(fft1) * norm(fft2))

print("Similarity Score:", similarity)
