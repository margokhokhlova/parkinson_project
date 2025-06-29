import numpy as np

class GaitDatasetNumpy:
    def __init__(self, PDL_features, ETL_features):
        # Ensure input arrays are float32
        self.PDL_features = np.array(PDL_features, dtype=np.float32)  # Shape: (n_pdl, n_feat, seq_len)
        self.ETL_features = np.array(ETL_features, dtype=np.float32)  # Shape: (n_etl, n_feat, seq_len)

        # Combine features
        self.features = np.concatenate((self.PDL_features, self.ETL_features), axis=0)

        # Create labels
        self.labels = np.concatenate((
            np.zeros(self.PDL_features.shape[0], dtype=np.int64),
            np.ones(self.ETL_features.shape[0], dtype=np.int64)
        ))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Transpose to (sequence_length, num_features) → from (features, time)
        return self.features[idx].T, self.labels[idx]  # Shape: (600, 14), label


        