import numpy as np
import librosa
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
    # 1. Load file audio
    y, sr = librosa.load(file_path, sr=None)

    # 2. Pre-emphasis
    y = pre_emphasis(y)

    # 3. FFT dan framing
    fft_magnitude, freqs = fft(y, sr)

    # 4. Hitung semua fitur
    features = []

    # Spectral Centroid
    centroid = spectral_centroid(fft_magnitude, freqs)
    features.extend(compute_statistics(centroid))

     # Spectral Center
    center = spectral_center(fft_magnitude, freqs)
    features.extend(compute_statistics(center))

    # Spectral Contrast (6 band)
    contrast = spectral_contrast(fft_magnitude)
    for i in range(contrast.shape[1]):  # 6 band
        features.extend(compute_statistics(contrast[:, i]))

    # Spectral Spread
    spread = spectral_spread(fft_magnitude, freqs, centroid)
    features.extend(compute_statistics(spread))

    # Spectral Skewness
    skewness = spectral_skewness(fft_magnitude, freqs, centroid, spread)
    features.extend(compute_statistics(skewness))

    # Spectral Kurtosis
    kurtosis = spectral_kurtosis(fft_magnitude, freqs, centroid, spread)
    features.extend(compute_statistics(kurtosis))

    # Spectral Flux
    flux = spectral_flux(fft_magnitude)
    features.extend(compute_statistics(flux))

    # Spectral Rolloff
    rolloff = spectral_rolloff(fft_magnitude, freqs)
    features.extend(compute_statistics(rolloff))

    # Spectral Flatness
    flatness = spectral_flatness(fft_magnitude)
    features.extend(compute_statistics(flatness))

    # Spectral Crest
    crest = spectral_crest_factor(fft_magnitude)
    features.extend(compute_statistics(crest))

    # Spectral Slope
    slope = spectral_slope(fft_magnitude, freqs)
    features.extend(compute_statistics(slope))    

    # Spectral Entropy
    entropy = spectral_entropy(fft_magnitude, freqs)
    features.extend(compute_statistics(entropy))    

    # Spectral Energy
    energy = spectral_energy(fft_magnitude)
    features.extend(compute_statistics(energy))

    # Spectral Roughness (frame-based, seperti contrast)
    roughness_per_frame = [spectral_roughness(fft_magnitude[i], freqs) for i in range(fft_magnitude.shape[0])]
    roughness_per_frame = np.array(roughness_per_frame) 
    features.extend(compute_statistics(np.array(roughness_per_frame)))

    return np.array(features).reshape(1, -1)
