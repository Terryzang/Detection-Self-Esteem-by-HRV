import numpy as np
import neurokit2 as nk
import pandas as pd
import os
import glob
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# File Pathway
file_path = 'E:\\'
target_folder = 'E:\\'
acq_files = glob.glob(os.path.join(file_path, '*.npy'))
sampling_rate = 2000

# Create a normalization instance
scaler = MinMaxScaler(feature_range=(-1, 1))

# Initialize a counter
processed_files_count = 0

# Initializes the feature list
all_features = []

# Display the progress bar using tqdm
for filename in tqdm(acq_files, desc="Processing files"):
    ecg_signal = np.load(filename)

    # Preprocess ECG
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method='neurokit')
    _,info = nk.ecg_peaks(ecg_cleaned,sampling_rate=sampling_rate, method='neurokit',correct_artifacts=False)

    # Extract R-wave position
    r_peaks = info["ECG_R_Peaks"]
    feature_dict = {
        "Filename": os.path.splitext(os.path.basename(filename))[0],
        **nk.hrv(r_peaks, sampling_rate=sampling_rate, show=False).iloc[0].to_dict(),
    }

    # Convert all features to scalars
    for key, value in feature_dict.items():
        if isinstance(value, (np.ndarray, pd.Series)):
            feature_dict[key] = value.item() if value.size == 1 else value
    all_features.append(feature_dict)

    processed_files_count += 1

# Convert all feature dictionaries to DataFrame
features_df = pd.DataFrame(all_features)

# Save the DataFrame to an Excel file
excel_filename = os.path.join(target_folder, '.xlsx')
features_df.to_excel(excel_filename, index=False)
print("Feature extraction completed and saved to Excel.")