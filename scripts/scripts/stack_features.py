import numpy as np
import torch

def stack_features(X_PDL_frequency, X_PDL_time, X_ETL_frequency,X_ETL_time):
    assert len(X_PDL_frequency) == len(X_PDL_time)
    assert len(X_ETL_frequency) == len(X_ETL_time)
    freq_PDL_array = np.array(X_PDL_frequency)
    time_PDL_array = np.array(X_PDL_time)

    # Stack along the new dimension
    X_PDL_features = np.stack((freq_PDL_array, time_PDL_array), axis=-1)

    # Check the shape of the result
    #print(X_PDL_features.shape)  # Should print (9, 600, 2)

    freq_ETL_array =  np.array(X_ETL_frequency)
    time_ETL_array =  np.array(X_ETL_time)

    X_ETL_features = np.stack((freq_ETL_array, time_ETL_array), axis=-1)

    #print(X_ETL_features.shape)  # Should print (9, 600, 2)

    X_PDL_flat = X_PDL_features.reshape(-1, 2)
    X_ETL_flat = X_ETL_features.reshape(-1, 2)

    # Create labels
    y_PDL = np.ones(X_PDL_flat.shape[0])
    y_ETL = np.zeros(X_ETL_flat.shape[0])

    # Combine features and labels
    X = np.vstack((X_PDL_flat, X_ETL_flat))
    y = np.concatenate((y_PDL, y_ETL))

    # Print the shapes to verify
    print(X.shape)  # Should print (600*9 + 600*13, 2)
    print(y.shape)  # Should print (600*9 + 600*13,)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor

   
def get_min_max_values(X_PDL_frequency, X_PDL_time, X_ETL_frequency, X_ETL_time):
    assert len(X_PDL_frequency) == len(X_PDL_time)
    assert len(X_ETL_frequency) == len(X_ETL_time)

    # Convert lists to numpy arrays
    freq_PDL_array = np.array(X_PDL_frequency)
    time_PDL_array = np.array(X_PDL_time)
    freq_ETL_array = np.array(X_ETL_frequency)
    time_ETL_array = np.array(X_ETL_time)

    # Stack the frequency and time arrays for both PDL and ETL
    all_freq = np.hstack((freq_PDL_array.flatten(), freq_ETL_array.flatten()))
    all_time = np.hstack((time_PDL_array.flatten(), time_ETL_array.flatten()))

    # Calculate min and max for frequency and time
    min_freq = np.min(all_freq)
    max_freq = np.max(all_freq)
    min_time = np.min(all_time)
    max_time = np.max(all_time)

    return min_freq, max_freq, min_time, max_time

def get_per_patient_features(X_PDL_frequency, X_PDL_time, X_ETL_frequency,X_ETL_time):
    freq_PDL_array = np.array(X_PDL_frequency)
    time_PDL_array = np.array(X_PDL_time)

    # Stack along the new dimension
    X_PDL_features = np.stack((freq_PDL_array, time_PDL_array), axis=-1)

    # Check the shape of the result
    print(X_PDL_features.shape)  # Should print (9, 600, 2)

    freq_ETL_array =  np.array(X_ETL_frequency)
    time_ETL_array =  np.array(X_ETL_time)

    X_ETL_features = np.stack((freq_ETL_array, time_ETL_array), axis=-1)
    return X_PDL_features, X_ETL_features