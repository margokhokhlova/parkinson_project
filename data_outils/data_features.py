import pandas as pd
import numpy as np
import pdb

feat_order = [
    'ampl',
    'BandwidthesHz',
    'Bottom',
    'flash_periods',
    'flash_seconds',
    'frac_band',
    'freq',
    'left',
    'phase',
    'Psi0',
    'right',
    'timeSignal',
    'top',
    'time_sincelast',
]

def get_feature(index_f, data_path = 'C:/Users/khokhlovam/Documents/kotelnikov/data/data_lstm_august24_PDET_left_right.csv', min_len_established = None, skip_patients = [55, 70,71,72,73,74,75,76]):
    df= pd.read_csv(data_path, header=None, names=range(2150)) # 
    #df.head(15)
    freq_valid = [-np.inf, +np.inf] # was previosly used to remove some frequences, no longer used but shouldn't be changed
    X_PDL= []
    X_ETL= []
    min_len= float('inf')  # Using infinity to start
    patient_i= 1
    for i in range(index_f,len(df),14): # 14 per item 
        label= df.iloc[[i]][0].values[0]    
        feat_range= df.iloc[[i]].values.tolist()[0] 
        patient_id =  df.iloc[[i]][1].values[0] 
        #print(patient_id)
        if patient_id in skip_patients:
            #print(f'skipping {patient_i}: {patient_id}')
            continue     
        valid_indexes= [f for f in range(3,len(feat_range)) if str(feat_range[f]) !='nan' and  feat_range[f]>=freq_valid[0] and feat_range[f]<=freq_valid[1]]
        feat= [feat_range[x] for x in valid_indexes]   
        if len(feat)<min_len:
            min_len= len(feat)
        if min_len_established == None:
            min_len_established = min_len
        if label == 'Left':
            X_PDL.append(feat[:min_len_established])
        elif label == 'ETLeft':
            X_ETL.append(feat[:min_len_established])
        #print(f'Patient{i} verification {label},  {feat_range[2]}')
        patient_i+= 1
    return X_PDL, X_ETL


class FeatNormalisation:
    def __init__(self, num_feat=14, min_len_established=600):
        self.min_values = None
        self.max_values = None
        self.num_feat = num_feat
        self.min_len_established = min_len_established

    def calculate_norm_values(self, all_train):
        if not isinstance(all_train, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        # all_train shape: (N, F, T)
        reshaped = all_train.transpose(1, 0, 2).reshape(self.num_feat, -1)  # Shape: (F, N*T)
        self.min_values = np.min(reshaped, axis=1, keepdims=True)  # Shape: (F, 1)
        self.max_values = np.max(reshaped, axis=1, keepdims=True)  # Shape: (F, 1)

    def normalize(self, features):
        if self.min_values is None or self.max_values is None:
            raise RuntimeError("Normalization values not calculated. Call `calculate_norm_values()` first.")
        if not isinstance(features, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        N, F, T = features.shape
        reshaped = features.transpose(1, 0, 2).reshape(self.num_feat, -1)  # Shape: (F, N*T)
        norm = (reshaped - self.min_values) / (self.max_values - self.min_values + 1e-8) - 0.5
        norm = norm.reshape(self.num_feat, N, self.min_len_established).transpose(1, 0, 2)  # Back to (N, F, T)
        return norm

def to_sktime_format(X_np):
    """Convert NumPy array (n_samples, n_timepoints, n_channels) to sktime multivariate nested format."""
    n_samples, n_timepoints, n_channels = X_np.shape
    nested_data = pd.DataFrame({
        f"var_{c}": [pd.Series(X_np[i, :, c]) for i in range(n_samples)]
        for c in range(n_channels)
    })
    return nested_data


__all__ = ['get_feature', 'feat_order','FeatNormalisation', 'to_sktime_format']