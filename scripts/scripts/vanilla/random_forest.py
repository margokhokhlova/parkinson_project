import yaml
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import  Dataset, DataLoader
import matplotlib.pyplot as plt
sys.path.append('scripts/scripts/')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd

class Features:
    def __init__(self, data_path):
        """Initialize the Features class by loading the dataset."""
        self.df = pd.read_csv(data_path, header=None, names=range(2150))

    def get_features_patient(self, patient_id, min_len_established=None):
        """Extract features for a specific patient.

        Args:
            patient_id (int): The patient identifier.
            min_len_established (int, optional): Column index to limit the feature extraction.

        Returns:
            tuple: (patient_features, label)
        """
        patient_data = self.df[self.df.iloc[:, 1] == patient_id]
        
        if patient_data.empty:
            raise ValueError(f"No data found for patient ID {patient_id}")

        label = patient_data.iloc[0, 0]  # Extract label from first column

        # Extract feature columns starting from index 4
        min_len_established = min_len_established or self.df.shape[1]  # Default to max columns if not specified
        patient_features = patient_data.iloc[:, 3:min_len_established]

        return patient_features.to_numpy(), label


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config



def main():
    # Load the configuration from the YAML file
    config = load_config('scripts/scripts/config.yaml')
    
    # Access the settings from the config
    learning_rate = config['settings']['learning_rate']
    batch_size = config['settings']['batch_size']
    num_epochs = config['settings']['num_epochs']
    model_name = config['settings']['model_name']
    data_path = config['settings']['data_path']
    combinations = config['settings']['possible_pairs']
    save_path = config['settings']['save_path']
    feature_names =  config['settings']['feat_order']
    
    # Print out the configuration
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Model Name: {model_name}")
    print(f"Data Path: {data_path}")
    print(f"Feature_names: {feature_names}")


    new_path=save_path+f'/vanilla_rnn'  
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    # Initialize lists to store the features and labels
    feat_inst = Features(data_path= data_path)
    excluded_values = {55, 70, 71, 72, 73, 74, 75, 76}  # Use a set for faster lookup
    filtered_patients = [i for i in range(41, 76) if i not in excluded_values]
    patients, labels = [], []
    for p in filtered_patients:   
        feat_patient_p, label_p = feat_inst.get_features_patient(p)
        # Extract the 14th feature (row index 13)
        feature_14 = feat_patient_p[13, :]
        # Find the last non-NaN value
        last_valid_index = np.where(~np.isnan(feature_14))[0][-2]  # Get one before last non-NaN index
        last_value = feature_14[last_valid_index]
        print(f"Patient {p} features  {feat_patient_p[:,:last_valid_index ].shape}, label {label_p},  last time since last {last_value}")
        #patients.append(feat_patient_p[:,:last_valid_index ])
        # Compute mean of all extracted features for this patient
        mean_features = np.nanmean(feat_patient_p[:, :last_valid_index], axis=1)  # Mean across time steps
        patients.append(mean_features)
        labels.append(label_p)

    # Convert lists to numpy arrays
    X = np.array(patients)  # Features
    y = np.array(labels)     # Labels

    # Encode labels if necessary (convert strings to numbers)
    label_mapping = {label: idx for idx, label in enumerate(set(y))}
    y_encoded = np.array([label_mapping[label] for label in y])

    # Store accuracy scores & feature importances
    accuracies = []
    feature_importance_sum = np.zeros(X.shape[1])

    # Repeat 100 times with different train-test splits
    for i in range(200):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.5, random_state=i, stratify=y_encoded
        )

        # Train a Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=i)
        clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        # Accumulate feature importance scores
        feature_importance_sum += clf.feature_importances_

        print(f"Iteration {i+1}: Accuracy = {accuracy:.2%}")

    # Compute mean accuracy over 100 iterations
    mean_accuracy = np.mean(accuracies)
    print(f"\nMean Random Forest Accuracy over 100 iterations: {mean_accuracy:.2%}")

    # Compute average feature importance
    feature_importance_avg = feature_importance_sum / 100

    # Get the most important features (sorted)
    feature_names = [
        "ampl", "BandwidthesHz", "Bottom", "flash_periods", "flash_seconds", 
        "frac_band", "freq", "left", "phase", "Psi0", "right", "timeSignal", "top", "time_sincelast"
    ]
    important_features = sorted(
        zip(feature_names, feature_importance_avg), key=lambda x: x[1], reverse=True
    )

    # Display feature importances
    print("\nFeature Importance Ranking:")
    for i, (feature, importance) in enumerate(important_features):
        print(f"{i+1}. {feature}: {importance:.4f}")

main()