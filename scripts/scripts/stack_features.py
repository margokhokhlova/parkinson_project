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



def stack_feature_lists(X_PDL_list, X_ETL_list):
    """
    Stacks features for 14 dimensions from PDL and ETL sources, and creates a corresponding label vector.
    
    Parameters:
        X_PDL_list (list): A list of arrays  from PDL patients.
        X_ETL_list (list): A list of arrays from ETL patients.
    
    Returns:
        X_tensor (torch.Tensor): Stacked features as a torch tensor (shape: [21600, 14]).
        y_tensor (torch.Tensor): Corresponding labels (shape: [21600]), where 1 indicates PDL and 0 indicates ETL.
    """
    
   
    # Stack the features for PDL and ETL separately
    stacked_PDL = np.stack(X_PDL_list, axis=-1)  # Shape: (600, 15, 14)
    stacked_ETL = np.stack(X_ETL_list, axis=-1)  # Shape: (600, 21, 14)

    # Flatten the features across all patients
    X_PDL_flat = stacked_PDL.reshape(15 * 600, 14) 
    X_ETL_flat = stacked_ETL.reshape(21 * 600, 14)
    
    # Create labels: 1 for PDL, 0 for ETL
    y_PDL = np.ones(X_PDL_flat.shape[0])  # Shape: (15 * 600,)
    y_ETL = np.zeros(X_ETL_flat.shape[0])  # Shape: (21 * 600,)
    
    # Stack the features and labels together
    X = np.vstack((X_PDL_flat, X_ETL_flat))  # Shape: (43200, 14)
    y = np.concatenate((y_PDL, y_ETL))  # Shape: (43200,)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor


def get_all_features_per_patient(PDL_features):
    # Assuming PDL_features is a list of 14 elements, each of shape [15, 600]
    # We want to transpose it to get 15 patients with their corresponding features (14, 600)

    # Convert the list of arrays into a single numpy array for easy manipulation
    PDL_features_array = np.array(PDL_features)  # Shape: [14, 15, 600]

    # Transpose the array to rearrange it to [15, 14, 600]
    # This way, we have 15 patients, and for each patient, an array of shape [14, 600]
    patients_with_features = PDL_features_array.transpose(1, 0, 2)
    return patients_with_features
