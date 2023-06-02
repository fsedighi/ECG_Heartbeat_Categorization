#Squeeze - resamples the signal along the time (horizontal) axis by shortening the distance between two consecutive samples.
#Stretch - resamples the signal along the time (horizontal) axis by elongating the signals.
#Amplify - amplifies the signal (along vertical axis) but still keeps the range between 0 and 1.
#Shrink - reduces the amplitude (along vertical axis) as well as squeezes the signal.

#These transformations are applied to every signal of the dataset and each of these transformations is saved and appended to the original dataset.
#These transformations in the signals are completely lossless transformations [23] and do not change the nature, quality and file size of the signals.

import numpy as np
import pandas as pd
import pywt

def squeeze_signal(signal, factor_range=(0.8, 1.2)):
    factor = np.random.uniform(*factor_range)
    if factor == 0:
        factor = 0.1  # or any other non-zero default value
    squeezed_signal = np.interp(np.arange(0, len(signal), factor), np.arange(len(signal)), signal)
    return squeezed_signal

def stretch_signal(signal, factor_range=(0.8, 1.2)):
    factor = np.random.uniform(*factor_range)
    if factor == 0:
        factor = 0.1  # or any other non-zero default value
    stretched_signal = np.interp(np.arange(0, len(signal), 1/factor), np.arange(len(signal)), signal)
    return stretched_signal

def amplify_signal(signal, amplification_range=(0.5, 1.5)):
    amplification = np.random.uniform(*amplification_range)
    amplified_signal = signal * amplification
    return amplified_signal

def shrink_signal(signal, shrinkage_range=(0.5, 1.0)):
    shrinkage = np.random.uniform(*shrinkage_range)
    shrinked_signal = signal * shrinkage
    return shrinked_signal

def augment_dataset(dataframe, augmentation_factor=1):
    augmented_data = []
    for _, row in dataframe.iterrows():
        signal = row.iloc[:-1].values
        label = row.iloc[-1]
        if label != 0:
            for _ in range(augmentation_factor):
                augmented_signal = apply_random_augmentation(signal)
                augmented_row = pd.Series(np.append(augmented_signal, label))
                augmented_data.append(row)
                augmented_data.append(augmented_row)
        elif label == 0:
            augmented_data.append(row)
    augmented_dataframe = pd.DataFrame(augmented_data, columns=dataframe.columns)
    augmented_dataframe = augmented_dataframe.reset_index(drop=True)
    return augmented_dataframe

def apply_random_augmentation(signal):
    # augmentation_functions = [squeeze_signal, stretch_signal, amplify_signal, shrink_signal]
    augmentation_functions = [amplify_signal, shrink_signal]
    augmentation_func = np.random.choice(augmentation_functions)
    augmented_signal = augmentation_func(signal)
    return augmented_signal

def add_noise(signal, level):
    noise = np.random.normal(0, level, size=len(signal))
    noisy_signal = signal + noise
    return noisy_signal

def add_noise_dataset(dataframe, level=0.02):
    noisy_data = []
    for _, row in dataframe.iterrows():
        signal = row.iloc[:-1].values
        label = row.iloc[-1]
        noisy_signal = add_noise(signal, level=level)
        noisy_row = pd.Series(np.append(noisy_signal, label))
        noisy_data.append(noisy_row)
    noisy_dataframe = pd.DataFrame(noisy_data, columns=dataframe.columns)
    return noisy_dataframe

def apply_denoising(signal, wavelet='db4', level=2):
    # Perform the Discrete Wavelet Transform
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Calculate the threshold using the median absolute deviation (MAD)
    mad = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = mad * np.sqrt(2 * np.log(len(signal)))
    # Apply soft thresholding to the DWT coefficients
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    # Reconstruct the denoised signal
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal

def denoise_dataset(dataframe, wavelet='db4', level=2):
    denoised_data = []
    for _, row in dataframe.iterrows():
        signal = row.iloc[:-1].values
        label = row.iloc[-1]
        denoised_signal = apply_denoising(signal, wavelet=wavelet, level=level)
        denoised_row = pd.Series(np.append(denoised_signal, label))
        denoised_data.append(denoised_row)
    denoised_dataframe = pd.DataFrame(denoised_data, columns=dataframe.columns)
    return denoised_dataframe

