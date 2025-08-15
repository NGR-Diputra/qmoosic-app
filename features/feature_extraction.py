import numpy as np
import soundfile as sf
from pydub import AudioSegment
import io
import os
from .functions import (
    fft,
    spectral_centroid, spectral_rolloff, spectral_spread, spectral_flatness,
    spectral_contrast, spectral_entropy, spectral_center, spectral_crest_factor,
    spectral_energy, spectral_flux, spectral_slope, spectral_roughness,
    spectral_skewness, spectral_kurtosis, compute_statistics
)

def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def feature_extraction(file_path):
    # 1. Load MP3 langsung ke memory → WAV
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mp3":
        audio = AudioSegment.from_file(file_path, format="mp3")
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)
        y, sr = sf.read(buf)
    else:
        y, sr = sf.read(file_path)

    # 2. Kalau stereo → ambil channel pertama
    if y.ndim > 1:
        y = y[:, 0]

    # 3. Pre-emphasis
    y = pre_emphasis(y)

    # 4. FFT dan framing
    fft_magnitude, freqs = fft(y, sr)

    # 5. Hitung semua fitur
    features = []

    centroid = spectral_centroid(fft_magnitude, freqs)
    features.extend(compute_statistics(centroid))

    center = spectral_center(fft_magnitude, freqs)
    features.extend(compute_statistics(center))

    contrast = spectral_contrast(fft_magnitude)
    for i in range(contrast.shape[1]):
        features.extend(compute_statistics(contrast[:, i]))

    spread = spectral_spread(fft_magnitude, freqs, centroid)
    features.extend(compute_statistics(spread))

    skewness = spectral_skewness(fft_magnitude, freqs, centroid, spread)
    features.extend(compute_statistics(skewness))

    kurtosis = spectral_kurtosis(fft_magnitude, freqs, centroid, spread)
    features.extend(compute_statistics(kurtosis))

    flux = spectral_flux(fft_magnitude)
    features.extend(compute_statistics(flux))

    rolloff = spectral_rolloff(fft_magnitude, freqs)
    features.extend(compute_statistics(rolloff))

    flatness = spectral_flatness(fft_magnitude)
    features.extend(compute_statistics(flatness))

    crest = spectral_crest_factor(fft_magnitude)
    features.extend(compute_statistics(crest))

    slope = spectral_slope(fft_magnitude, freqs)
    features.extend(compute_statistics(slope))

    entropy = spectral_entropy(fft_magnitude, freqs)
    features.extend(compute_statistics(entropy))

    energy = spectral_energy(fft_magnitude)
    features.extend(compute_statistics(energy))

    roughness_per_frame = [
        spectral_roughness(fft_magnitude[i], freqs)
        for i in range(fft_magnitude.shape[0])
    ]
    roughness_per_frame = np.array(roughness_per_frame)
    features.extend(compute_statistics(roughness_per_frame))

    return np.array(features).reshape(1, -1)
