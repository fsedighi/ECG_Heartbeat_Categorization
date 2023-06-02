import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from scipy.signal import stft
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# The effectiveness of feature selection techniques, including PCA, Univariate Selection,
# Recursive Feature Elimination (RFE), Correlation Analysis, can vary depending on the specific dataset,
# the nature of the problem, and the algorithms being used.


def recursive_feature_elimination(X, y, k, estimator = LinearRegression()):
    # Apply RFE with the specified estimator and select top k features
    selector = RFE(estimator, n_features_to_select=k)
    selector.fit(X, y)
    X_new = selector.transform(X)
    return X_new, selector.support_

def correlation_analysis(X, y, threshold = 0.3): 
    # Calculate correlation between features and target variable
    df = pd.concat([X, y], axis=1)
    corr_matrix = df.corr()
    corr_with_target = corr_matrix.iloc[:-1, -1]  
    # Select features based on correlation threshold
    relevant_features = corr_with_target[abs(corr_with_target) > threshold].index
    X_new = X[relevant_features]
    return X_new, relevant_features


def time_frequency_features(X, method = 'wigner_ville'):   
    if method == 'wigner_ville':
        transformed_signals = wigner_ville(X)
    elif method == 'stft':
        transformed_signals = stft(X)
    # Extract features from the WVD DataFrame
    features = []
    for i in range(transformed_signals.shape[0]):
        signal_features = []      
        # feature extraction
        
        # Feature 1: Spectral Power
        power_spectrum = np.square(np.abs(transformed_signals.iloc[i, :]))
        total_power = np.sum(power_spectrum)
        signal_features.append(total_power)
        
        # Feature 2: Spectral Entropy
        normalized_power_spectrum = power_spectrum / np.sum(power_spectrum)
        entropy = -np.sum(normalized_power_spectrum * np.log2(normalized_power_spectrum))
        signal_features.append(entropy)
        
        # Feature 3: Spectral Flatness
        geometric_mean = np.exp(np.mean(np.log(power_spectrum + np.finfo(float).eps)))
        arithmetic_mean = np.mean(power_spectrum)
        flatness = geometric_mean / arithmetic_mean
        signal_features.append(flatness)
        
        # Feature 4: Spectral Edge Frequency
        cumulative_power = np.cumsum(power_spectrum)
        edge_frequency = np.argmax(cumulative_power >= 0.95 * total_power)
        signal_features.append(edge_frequency)
        
        # Feature 5: Peak Frequency
        peak_frequency = np.argmax(power_spectrum)
        signal_features.append(peak_frequency)
  
    # Create a new DataFrame with the extracted features and class labels
    X_new = pd.DataFrame(features, columns=['Spectral Power', 'Spectral Entropy', 'Spectral Flatness', 'Spectral Edge Frequency', 'Peak Frequency'])  # Add column names as desired
    return X_new    



def stft(signals, fs=360, nperseg=256, noverlap=128):
    # Apply STFT to each signal
    stft_results = []
    for signal in signals:
        f, t, stft_data = stft(signal, fs=fs, window='hamming', nperseg=nperseg, noverlap=noverlap)
        stft_results.append(stft_data)

    # Convert the STFT results to a NumPy array
    stft_results = np.array(stft_results)

    # Create a DataFrame from the STFT results
    signals_new = pd.DataFrame(stft_results)

    # Return the DataFrame
    return signals_new



def wigner_ville(signals, fs=360, nperseg=256, noverlap=128):
        
    # Apply Wigner-Ville distribution to each ECG signal
    wvd_results = []
    for signal in signals:
        f, t, wvd = spectrogram(signal, fs=fs, window='hamming', nperseg=nperseg, noverlap=noverlap)
        wvd_results.append(wvd)
    
    # Convert the WVD results to a NumPy array
    wvd_results = np.array(wvd_results)
    
    # Create a DataFrame from the WVD results
    signals_new = pd.DataFrame(wvd_results)

    # Return the DataFrame
    return signals_new

