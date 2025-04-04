import numpy as np
from scipy.signal import welch
import scipy.signal as signal
import pywt, wfdb, time
from scipy.stats import skew, kurtosis
from statsmodels.tsa.ar_model import AutoReg
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def find_fs(signal, hours):
    return len(signal)/(hours*3600)

def notch_filter(data, fs, freq=50, quality=30):
    b, a = signal.iirnotch(freq, quality, fs)
    return signal.filtfilt(b, a, data)

def bandpass_filter(data, fs, lowcut=0.5, highcut=45, order=4):
    b, a = signal.butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    return signal.filtfilt(b, a, data)


def preprocess_signals(raw_signals, fs):
    preprocessed_signals = []
    for signal in raw_signals:
        # Apply notch filter
        filtered_signal = notch_filter(signal, fs)
        # Apply bandpass filter
        filtered_signal = bandpass_filter(filtered_signal, fs)
        preprocessed_signals.append(filtered_signal)
    return preprocessed_signals

def segment_signals(preprocessed_signals, fs, epoch_duration=30):
    samples_per_epoch = int(fs * epoch_duration)  
    num_epochs = preprocessed_signals.shape[1] // samples_per_epoch 

    truncated_signals = preprocessed_signals[:, :num_epochs * samples_per_epoch]

    segmented_signals = []
    for i in range(num_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        epoch = truncated_signals[:, start:end].flatten()  
        segmented_signals.append(epoch)

    return np.array(segmented_signals)  


def calculate_entropy(probabilities, entropy_type='shannon', q=2):
    if entropy_type == 'shannon':
        return -np.sum(probabilities * np.log(probabilities + 1e-10))  # Add small value to avoid log(0)
    elif entropy_type == 'renyi':
        return 1 / (1 - q) * np.log(np.sum(probabilities**q))
    elif entropy_type == 'tsallis':
        return 1 / (q - 1) * (1 - np.sum(probabilities**q))
    else:
        raise ValueError("Invalid entropy type. Choose 'shannon', 'renyi', or 'tsallis'.")
    
def transform_features(features):
    transformed_features = {}
    for key, value in features.items():
        # Apply transformation only if the value is non-negative
        if isinstance(value, (int, float)) and value >= 0:
            transformed_features[key] = np.arcsin(np.sqrt(value)) if value <= 1 else value
        else:
            # Keep the original value for non-numeric or invalid cases
            transformed_features[key] = value
    return transformed_features


def extract_modwt_features(signal, fs, wavelet='db4', level=5, q=2, ar_order=3):
    """
    Perform wavelet decomposition (DWT) and extract statistical features.
    
    Parameters:
        signal (array): Input signal (e.g., EEG, EOG, EMG).
        fs (float): Sampling frequency of the signal.
        wavelet (str): Wavelet type (default: 'db4').
        level (int): Decomposition depth (default: 6).
    
    Returns:
        features (dict): Extracted statistical features for each decomposition level.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode='periodization')
    
    freq_ranges = {
        'D1': (fs / 2**1, fs / 2**2),
        'D2': (fs / 2**2, fs / 2**3),
        'D3': (fs / 2**3, fs / 2**4),
        'D4': (fs / 2**4, fs / 2**5),
        'D5': (fs / 2**5, fs / 2**6),
        'A5': (0, fs / 2**6)
    }
    
    features = {}
    total_power = 0
    slow_wave_power = 0

    for coeff in coeffs:
        total_power += np.sum(coeff**2)

    for i, coeff in enumerate(coeffs):
        level_name = f"D{i+1}" if i < level else f"A{i}"
        power = np.sum(coeff**2) 
        features[f'{level_name}_mean'] = np.mean(coeff)
        features[f'{level_name}_std'] = np.std(coeff)
        features[f'{level_name}_energy'] = np.sum(coeff**2)
        features[f'{level_name}_entropy'] = -np.sum(coeff * np.log(np.abs(coeff) + 1e-10))
        features[f'{level_name}_range'] = np.max(coeff) - np.min(coeff)
        features[f'{level_name}_percentage_energy'] = (power / total_power) * 100 if total_power > 0 else 0

        hist, _ = np.histogram(coeff, bins=50, density=True) 
        probabilities = hist / np.sum(hist)  
        features[f'{level_name}_shannon_entropy'] = calculate_entropy(probabilities, entropy_type='shannon')
        features[f'{level_name}_renyi_entropy'] = calculate_entropy(probabilities, entropy_type='renyi', q=q)
        features[f'{level_name}_tsallis_entropy'] = calculate_entropy(probabilities, entropy_type='tsallis', q=q)
        features[f'{level_name}_relative'] = freq_ranges[level_name]

        if level_name in ['D4', 'D5', 'A5']:
            slow_wave_power += power

        f, pxx = welch(coeff, fs=fs, nperseg=min(len(coeff), 256))  
        f_l, f_h = freq_ranges[level_name]  
        band_mask = (f >= f_l) & (f <= f_h)  
        f_band = f[band_mask]
        pxx_band = pxx[band_mask]
        
        if len(f_band) > 0 and np.sum(pxx_band) > 0:
            # Center frequency (fc)
            fc = np.sum(f_band * pxx_band) / np.sum(pxx_band)
            # Bandwidth (fr)
            fr = np.sqrt(np.sum(((f_band - fc)**2) * pxx_band) / np.sum(pxx_band))
            # Spectral value at center frequency (Sfc)
            sfc = pxx[np.argmin(np.abs(f - fc))]
        else:
            fc, fr, sfc = 0, 0, 0  # Default values if no valid band
        features[f'{level_name}_center_frequency'] = fc
        features[f'{level_name}_bandwidth'] = fr
        features[f'{level_name}_spectral_value_at_fc'] = sfc

        variance = np.var(coeff)
        first_derivative = np.diff(coeff)
        second_derivative = np.diff(first_derivative)
        hjorth_activity = variance
        hjorth_mobility = np.sqrt(np.var(first_derivative) / variance) if variance > 0 else 0
        hjorth_complexity = (
            np.sqrt(np.var(second_derivative) / np.var(first_derivative)) / hjorth_mobility
            if hjorth_mobility > 0 and np.var(first_derivative) > 0
            else 0
        )
        features[f'{level_name}_hjorth_activity'] = hjorth_activity
        features[f'{level_name}_hjorth_mobility'] = hjorth_mobility
        features[f'{level_name}_hjorth_complexity'] = hjorth_complexity
        
        features[f'{level_name}_skewness'] = skew(coeff)
        features[f'{level_name}_kurtosis'] = kurtosis(coeff)

        try:
            ar_model = AutoReg(coeff, lags=ar_order, old_names=False).fit()
            for j, ar_coeff in enumerate(ar_model.params):
                features[f'{level_name}_ar_coeff_{j+1}'] = ar_coeff
        except Exception:
            for j in range(ar_order):
                features[f'{level_name}_ar_coeff_{j+1}'] = 0  # Default to 0 if AR model fails

        features[f'{level_name}_percentile_25'] = np.percentile(coeff, 25)
        features[f'{level_name}_percentile_50'] = np.percentile(coeff, 50)  
        features[f'{level_name}_percentile_75'] = np.percentile(coeff, 75)
        

    features['slow_wave_index'] = slow_wave_power/total_power if total_power > 0 else 0
    transformed_features = transform_features(features)
    return transformed_features

def extract_features(segmented_signals, fs):
    features_list = []
    print("Extracting features...")
    for epoch in segmented_signals:
        features = extract_modwt_features(epoch, fs, level=5)  
        for key, value in features.items():
            if not np.isscalar(value):  
                features[key] = np.mean(value)  
        features_list.append(features)

    print("Ensuring consistent feature dimensions...")
    all_keys = set().union(*(features.keys() for features in features_list)) 
    for features in features_list:
        for key in all_keys:
            if key not in features:
                features[key] = 0 # Default value for missing features

    return features_list

def normalize_features(features_list):
    all_features = {}
    for features in features_list:
        for key, value in features.items():
            if key not in all_features:
                all_features[key] = []
            all_features[key].append(value)
    
    normalized_features_list = []
    for features in features_list:
        normalized_features = {}
        for key, value in features.items():
            feature_values = all_features[key]
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            if max_val > min_val:  
                normalized_features[key] = (value - min_val) / (max_val - min_val)
            else:
                normalized_features[key] = 0  
        normalized_features_list.append(normalized_features)
    
    return normalized_features_list

def filter_features(features, labels, threshold=0.01):
    mi_scores = mutual_info_classif(features, labels)
    
    selected_indices = np.where(mi_scores > threshold)[0]
    filtered_features = features[:, selected_indices]
    
    return filtered_features, selected_indices

def mrmr_selection(features, labels, k=10):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selected_features = selector.fit_transform(features, labels)
    selected_indices = selector.get_support(indices=True)
    
    return selected_features, selected_indices

def train_and_evaluate_binary_svm(selected_features, labels):
    X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=0.2, random_state=42, stratify=labels)
    
    print("Training SVM...")
    start_time = time.time()
    svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    end_time = time.time()
    print(f"SVM training completed in {end_time - start_time:.2f} seconds.")

    print("Evaluating SVM...")
    y_pred = svm_model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    unique_classes = np.unique(y_test)
    target_names = ['Wake', 'Sleep']
    adjusted_target_names = [target_names[i] for i in unique_classes]

    print(classification_report(y_test, y_pred, target_names=adjusted_target_names))
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

def train_and_evaluate_multiclass_svm(selected_features, labels):
    X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=0.2, random_state=42, stratify=labels)

    print("Training SVM...")
    start_time = time.time()
    svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    end_time = time.time()
    print(f"SVM training completed in {end_time - start_time:.2f} seconds.")

    print("Evaluating SVM...")
    y_pred = svm_model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    unique_classes = np.unique(y_test)
    target_names = ['Wake', 'NREM1', 'NREM2', 'NREM3', 'REM']  
    adjusted_target_names = [target_names[i] for i in unique_classes]

    print(classification_report(y_test, y_pred, target_names=adjusted_target_names))
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

def main_pipeline(raw_signals, labels, fs, type):
    # Step 1: Preprocess the signals
    print("Preprocessing signals...")
    preprocessed_signals = preprocess_signals(raw_signals, fs)
    preprocessed_signals = np.array(preprocessed_signals) 
    print("Preprocessed signals shape:", preprocessed_signals.shape)
    print("Preprocessing completed.")

    # Step 2: Segment the signals into 30-second epochs
    print("Segmentation started...")
    segmented_signals = segment_signals(preprocessed_signals, fs)
    print("Segmented signals shape:", segmented_signals.shape)
    print("Number of labels:", len(labels))

    # Ensure alignment between segmented signals and labels
    if segmented_signals.shape[0] != len(labels):
        raise ValueError(f"Mismatch: {segmented_signals.shape[0]} epochs, {len(labels)} labels")

    # Step 3: Extract features for each epoch
    print("Feature extraction started...")
    features_list = extract_features(segmented_signals, fs)
    print(f"Number of feature dictionaries: {len(features_list)}")

    # Step 4: Normalize features
    print("Feature extraction completed.")
    print("Normalizing features...")
    normalized_features_list = normalize_features(features_list)
    print(f"Number of normalized feature dictionaries: {len(normalized_features_list)}")

    # Step 5: Convert features to a matrix
    print("Feature normalization completed.")
    print("Converting features to matrix...")
    feature_matrix = np.array([list(features.values()) for features in normalized_features_list])
    print(f"Feature matrix shape: {feature_matrix.shape}")

    # Step 6: Filter features based on mutual information
    print("Feature conversion completed.")
    print("Filtering features based on mutual information...")
    filtered_features, _ = filter_features(feature_matrix, labels)

    # Step 7: Perform mRMR feature selection
    print("Feature filtering completed.")
    print("Performing mRMR feature selection...")
    selected_features, _ = mrmr_selection(filtered_features, labels, k=10)

    # Step 8: Train and evaluate the SVM classifier
    print("mRMR feature selection completed.")
    print("Training and evaluating SVM...")
    if type=="binary":
        train_and_evaluate_binary_svm(selected_features, labels)
    elif type=="multiclass":
        train_and_evaluate_multiclass_svm(selected_features, labels)
    print("SVM training and evaluation completed.")
    print("Pipeline completed.")
