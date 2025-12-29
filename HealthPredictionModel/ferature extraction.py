import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, hilbert
from scipy.stats import kurtosis, skew, entropy
from scipy.fft import fft
from sklearn.ensemble import RandomForestRegressor

# --- Settings ---
data_dir = r"BearingData_csv"
output_file = r"all_features.csv"
sampling_rate = 97656  # Hz
segment_duration = 1  # seconds
segment_size = sampling_rate * segment_duration

# --- Feature extraction ---
def extract_features(vibration_segment, sampling_rate):
    rms = np.sqrt(np.mean(vibration_segment**2))
    peak = np.max(np.abs(vibration_segment))

    features = {
        "rms": rms,
        "kurtosis": kurtosis(vibration_segment),
        "peak": peak,
        "crest_factor": peak / rms if rms !=0 else 0,
        "skewness": skew(vibration_segment),
        "std_dev": np.std(vibration_segment),
        "peak_to_peak": np.max(vibration_segment) - np.min(vibration_segment),
    }

    spectrum = np.abs(fft(vibration_segment))[:len(vibration_segment)//2]
    features["fft_peak"] = np.max(spectrum)
    features["spectral_centroid"] = np.sum(spectrum * np.arange(len(spectrum))) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0

    # Band Energy Ratio
    low_band_start = 50
    low_band_end = 150
    high_band_start = 300
    high_band_end = 400

    freqs = np.fft.fftfreq(len(vibration_segment), 1/sampling_rate)
    positive_freq_indices = np.where(freqs >= 0)[0]
    freqs = freqs[positive_freq_indices]
    spectrum_positive = spectrum[:len(positive_freq_indices)]

    low_band_indices = np.where((freqs >= low_band_start) & (freqs <= low_band_end))[0]
    high_band_indices = np.where((freqs >= high_band_start) & (freqs <= high_band_end))[0]

    low_band_energy = np.sum(spectrum_positive[low_band_indices]**2)
    high_band_energy = np.sum(spectrum_positive[high_band_indices]**2)

    features["band_energy_ratio"] = low_band_energy / high_band_energy if high_band_energy != 0 else 0

    # Amplitude Envelope Features
    analytic_signal = hilbert(vibration_segment)
    amplitude_envelope = np.abs(analytic_signal)
    fft_amplitude_envelope = np.fft.fft(amplitude_envelope)
    features["envelope_spectrum_peak"] = np.max(np.abs(fft_amplitude_envelope[1:len(fft_amplitude_envelope)//2]))

    features["entropy"] = entropy(np.histogram(vibration_segment, bins=50)[0])
    return features

# --- Main loop over all files ---
all_data = []

for day in range(1, 51):
    try:
        vib_file = os.path.join(data_dir, f"file{day}_vibration.csv")
        tach_file = os.path.join(data_dir, f"file{day}_tachometer.csv")  # not used

        vib_df = pd.read_csv(vib_file)
        vib_data = vib_df["Vibration"].values

        segment_features = []  # Accumulate features for each segment within a day

        for i in range(0, len(vib_data), segment_size):
            segment = vib_data[i:i + segment_size]
            if len(segment) < segment_size:
                continue

            features = extract_features(segment, sampling_rate)
            features["segment"] = i // segment_size
            features["day"] = day
            features["severity"] = (day - 1) / 49  # Normalized target

            all_data.append(features)
            segment_features.append(features)  # Add to the list for health indicator calculation
            print(all_data[-1])  # Print the last feature set for debugging

        print(f"✅ Processed Day {day}")

    except Exception as e:
        print(f"❌ Error on day {day}: {e}")

# --- Create DataFrame and calculate health indicator ---
final_df = pd.DataFrame(all_data)

# Group by day and compute mean values
grouped = final_df.groupby('day').agg({
    'kurtosis': 'mean',
    'envelope_spectrum_peak': 'mean'
}).reset_index()

# Smooth both signals using rolling mean
grouped['kurtosis_smoothed'] = grouped['kurtosis'].rolling(window=3, min_periods=1).mean()
grouped['envelope_smoothed'] = grouped['envelope_spectrum_peak'].rolling(window=3, min_periods=1).mean()

# Define weights
w1 = 0.5  # weight for kurtosis
w2 = 0.5  # weight for envelope spectrum peak

# Normalize features to [0,1] before combining (important!)
kurtosis_norm = (grouped['kurtosis_smoothed'] - grouped['kurtosis_smoothed'].min()) / (grouped['kurtosis_smoothed'].max() - grouped['kurtosis_smoothed'].min())
envelope_norm = (grouped['envelope_smoothed'] - grouped['envelope_smoothed'].min()) / (grouped['envelope_smoothed'].max() - grouped['envelope_smoothed'].min())

# Compute health indicator
grouped['health_indicator'] = w1 * kurtosis_norm + w2 * envelope_norm

# Merge health indicator back into the main DataFrame
final_df = pd.merge(final_df, grouped[['day', 'health_indicator']], on='day', how='left')


# --- Save extracted features ---
final_df.to_csv(output_file, index=False)
print(f"📁 All features saved to: {output_file}")

# --- Plot feature distributions ---
feature_columns = [col for col in final_df.columns if col not in ["segment", "day", "severity", "health_indicator"]]

plt.figure(figsize=(18, len(feature_columns)*2))
for idx, feature in enumerate(feature_columns, 1):
    plt.subplot(len(feature_columns)//3 + 1, 3, idx)
    plt.hist(final_df[feature], bins=50, alpha=0.7, color='blue')
    plt.title(feature)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --- Train Random Forest and plot feature importances ---
X = final_df[feature_columns]
y = final_df["severity"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_indices], align='center')
plt.xticks(range(len(importances)), [feature_columns[i] for i in sorted_indices], rotation=45, ha='right')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Relative Importance")
plt.tight_layout()
plt.show()

# --- Plot Feature Changes Over Days ---
plt.figure(figsize=(16, 10))
num_features = len(feature_columns)
num_rows = int(np.ceil(num_features / 3))  # Adjust layout as needed
for idx, feature in enumerate(feature_columns, 1):
    plt.subplot(num_rows, 3, idx)
    
    # Group by day and calculate the mean feature value for each day
    mean_feature_by_day = final_df.groupby("day")[feature].mean()
    
    plt.plot(mean_feature_by_day.index, mean_feature_by_day.values, marker='o', linestyle='-')
    plt.title(f"Mean {feature} Over Days")
    plt.xlabel("Day")
    plt.ylabel(feature)
    plt.grid(True)

plt.tight_layout()
plt.show()

# --- Plot Health Indicator Over Days ---
plt.figure(figsize=(10, 4))
plt.plot(final_df.groupby("day")["health_indicator"].mean().index, final_df.groupby("day")["health_indicator"].mean().values, color='purple', marker='o', label='Health Indicator')
plt.xlabel('Day')
plt.ylabel('Health Indicator (normalized)')
plt.title('Health Indicator vs. Day')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()