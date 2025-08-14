import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import scipy

"""
================================================================================
PEMROSESAN SINYAL MUSIK
================================================================================

"""

def fft(signal, sr, frame_size=1024, hop_size=512):

    # Windowing
    window = np.hamming(frame_size)

    fft_magnitudes = []

    # Framing Signal
    for start in range(0, len(signal) - frame_size, hop_size):
        frame = signal[start:start + frame_size] * window  # Windowing
        fft_frame = np.fft.fft(frame)

        # Spektrum
        magnitude_spectrum = np.abs(fft_frame[:frame_size // 2])  
        fft_magnitudes.append(magnitude_spectrum)

    fft_magnitudes = np.array(fft_magnitudes)

    # Bin Frekuensi
    freqs = np.fft.fftfreq(frame_size, 1 / sr)[:frame_size // 2]

    return fft_magnitudes, freqs


"""
================================================================================
FUNCTION EKSTRAKSI FITUR SPEKTRAL
================================================================================

"""

# Spectral Centtroid
def spectral_centroid(magnitude, freqs):

    magnitude_sums = np.sum(magnitude, axis=1, keepdims=True)
    magnitude_sums[magnitude_sums == 0] = 1e-10

    # Weighted sum of frequencies
    weighted_sum = np.dot(magnitude, freqs)

    # Calculate centroids
    centroids = weighted_sum / magnitude_sums.flatten()

    return centroids


# Spectral Rolloff
def spectral_rolloff(magnitude, freqs, rolloff_percentage=0.90):

    total_energy = np.sum(magnitude, axis=1, keepdims=True)

    # Cumulative sum of magnitude along frequency axis
    cumulative_energy = np.cumsum(magnitude, axis=1)

    rolloff_threshold = rolloff_percentage * total_energy
    rolloff_indices = np.argmax(cumulative_energy >= rolloff_threshold, axis=1)
    rolloff_frequencies = freqs[rolloff_indices]

    return rolloff_frequencies


# Specttral Spread
def spectral_spread(fft_magnitudes, freqs, spectral_centroids):

    # Distribution of frequency bins
    deviation = freqs - spectral_centroids[:, None]

    # Sum of magnitude for each frame-s
    magnitude_sums = np.sum(fft_magnitudes, axis=1)
    magnitude_sums[magnitude_sums == 0] = 1e-10

    # Calculate Spread
    weighted_variance = np.sum(fft_magnitudes * (deviation ** 2), axis=1) / magnitude_sums
    spread = np.sqrt(weighted_variance)

    return spread


# Spectral Contrast
def spectral_contrast(fft_magnitude, num_bands=6):
    fft_magnitude = np.array(fft_magnitude)

    # Convert to dB scale
    fft_magnitude = 20 * np.log10(fft_magnitude + 1e-10)

    num_freq_bins = fft_magnitude.shape[1]
    band_edges = np.logspace(np.log10(1), np.log10(num_freq_bins), num_bands + 1).astype(int)
    band_edges[0] = 0

    contrast = np.zeros((fft_magnitude.shape[0], num_bands))

    for frame_idx in range(fft_magnitude.shape[0]):
        for band_idx in range(num_bands):
            start_bin = band_edges[band_idx]
            end_bin = min(band_edges[band_idx + 1], num_freq_bins)

            band_magnitudes = fft_magnitude[frame_idx, start_bin:end_bin]
            if len(band_magnitudes) == 0:
                continue

            peak_threshold = np.percentile(band_magnitudes, 95)
            valley_threshold = np.percentile(band_magnitudes, 5)

            peaks = band_magnitudes[band_magnitudes >= peak_threshold]
            valleys = band_magnitudes[band_magnitudes <= valley_threshold]

            if len(peaks) > 0 and len(valleys) > 0:
                contrast[frame_idx, band_idx] = np.mean(peaks) - np.mean(valleys)

    return contrast


# Spectral Flatness
def spectral_flatness(fft_magnitude):
   
    # Convert magnitude spectrum to a numpy array
    fft_magnitude = np.array(fft_magnitude)

    # Calculate the geometric mean and arithmetic mean for each frame
    geometric_mean = np.exp(np.mean(np.log(fft_magnitude + 1e-10), axis=1))  # Add small value to avoid log(0)
    arithmetic_mean = np.mean(fft_magnitude, axis=1)

    # Calculate spectral flatness for each frame
    flatness = geometric_mean / (arithmetic_mean + 1e-10)  # Add small value to avoid division by zero
    return flatness


# Spectral Crest Factor
def spectral_crest_factor(fft_magnitude):
    
    # Calculate peak and RMS values
    peak = np.max(fft_magnitude, axis=1)
    rms = np.sqrt(np.mean(fft_magnitude**2, axis=1))

    # Calculate crest factor
    crest_factor = peak / (rms + 1e-10)  # Add small value to avoid division by zero
    return crest_factor


# Spectral Slope
def spectral_slope(fft_magnitude, freqs, use_log=True):
  
    # Ensure frequency and magnitude arrays are 2D (for multi-frame support)
    if fft_magnitude.ndim == 1:
        fft_magnitude = fft_magnitude[np.newaxis, :]  # Convert to (1, bins)

    # Log transformation (if enabled)
    if use_log:
        eps = 1e-10
        log_freqs = np.log1p(freqs)  # log(f + 1) to prevent log(0)
        log_magnitude = 10 * np.log10(fft_magnitude + eps)  # Convert to dB scale
    else:
        log_freqs = freqs
        log_magnitude = fft_magnitude

    # Compute means
    mean_freq = np.mean(log_freqs)
    mean_magnitude = np.mean(log_magnitude, axis=1, keepdims=True)

    # Compute numerator: weighted sum of (freqs - mean_freq) * (magnitude - mean_magnitude)
    numerator = np.sum((log_freqs - mean_freq) * (log_magnitude - mean_magnitude), axis=1)

    # Compute denominator: sum of squared frequency deviations
    denominator = np.sum((log_freqs - mean_freq) ** 2)

    # Avoid division by zero
    slope = np.divide(numerator, denominator, where=denominator != 0, out=np.zeros_like(numerator))

    return slope


# Spectral Flux
def spectral_flux(fft_magnitude):
   
    # Compute frame-to-frame difference
    diff = np.diff(fft_magnitude, axis=0)

    # Half-wave rectification: keep only positive changes
    diff = np.maximum(diff, 0)

    # Compute flux as the sum of squared differences
    flux = np.sqrt(np.sum(diff**2, axis=1))

    return flux


# Spectral Entropy
def spectral_entropy(fft_magnitude: np.ndarray, freq_bins: np.ndarray) -> np.ndarray:
    
    power_spectrum = fft_magnitude ** 2
    psd = power_spectrum / (np.sum(power_spectrum, axis=1, keepdims=True) + 1e-12)  # Normalize to probability distribution
    psd = np.clip(psd, 1e-12, 1)  # Avoid log(0)


    entropy_values = -np.sum(psd * np.log(psd), axis=1)  # Compute entropy per frame

    # Normalize entropy
    freq_range = freq_bins[-1] - freq_bins[0]
    if freq_range > 0:
        entropy_values /= np.log(freq_range)

    return entropy_values


# Spectral Center
def spectral_center(magnitudes, freqs):
    spectral_centers = []

    for magnitude in magnitudes:
        cumulative_magnitude = np.cumsum(magnitude)

        if cumulative_magnitude.size == 0 or cumulative_magnitude[-1] == 0:
            spectral_centers.append(0)  # Assign a default value
        else:
            normalized_energy = cumulative_magnitude / cumulative_magnitude[-1]
            spectral_center_idx = np.searchsorted(normalized_energy, 0.5)
            spectral_centers.append(freqs[min(spectral_center_idx, len(freqs) - 1)])

    return np.array(spectral_centers)


# Spectral Energy
def spectral_energy(fft_magnitude: np.ndarray) -> np.ndarray:
    return np.sum(fft_magnitude ** 2, axis=1)

# Spectral Roughness
def spectral_roughness(fft_magnitude, freq_bins, alpha=0.25, prominence=0.05):
   
    # Step 1: Find spectral peaks (magnitude and frequency)
    peaks, _ = find_peaks(fft_magnitude, prominence=prominence * np.max(fft_magnitude))

    if len(peaks) < 2:
        return 0  # No roughness if less than 2 peaks

    peak_freqs = freq_bins[peaks]
    peak_mags = fft_magnitude[peaks]

    # Step 2: Compute roughness for all peak pairs
    roughness = 0
    num_peaks = len(peak_freqs)

    for i in range(num_peaks):
        for j in range(i + 1, num_peaks):
            delta_f = abs(peak_freqs[i] - peak_freqs[j])
            roughness += np.exp(-alpha * delta_f) * (peak_mags[i] * peak_mags[j])

    return roughness


# Spectral Skewness
def spectral_skewness(fft_magnitudes, freqs, spectral_centroids, spectral_spreads):

    deviation = freqs - spectral_centroids[:, None]
    magnitude_sums = np.sum(fft_magnitudes, axis=1)

    magnitude_sums[magnitude_sums == 0] = 1e-10
    spectral_spreads[spectral_spreads == 0] = 1e-10

    # Third moment
    third_moment = np.sum(fft_magnitudes * (deviation ** 3), axis=1) / magnitude_sums

    # Spectral skewness
    skewness = third_moment / (spectral_spreads ** 3)

    return skewness


# Spectral Kurtosis
def spectral_kurtosis(fft_magnitudes, freqs, spectral_centroids, spectral_spreads):

    deviation = freqs - spectral_centroids[:, None]
    magnitude_sums = np.sum(fft_magnitudes, axis=1)

    magnitude_sums[magnitude_sums == 0] = 1e-10
    spectral_spreads[spectral_spreads == 0] = 1e-10

    # Calculate Kurtosis
    fourth_moment = np.sum(fft_magnitudes * (deviation ** 4), axis=1) / magnitude_sums

    # Spectral kurtosis
    kurtosis = fourth_moment / (spectral_spreads ** 4)

    return kurtosis


"""
================================================================================
APLIKASI STATISTIKA DESKRIPTIF
================================================================================

"""
def compute_statistics(feature):
    return [
        np.mean(feature), np.median(feature), np.var(feature), np.std(feature),
        np.min(feature), np.max(feature), scipy.stats.iqr(feature),
        scipy.stats.skew(feature), scipy.stats.kurtosis(feature)
    ]