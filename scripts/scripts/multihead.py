import yaml
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models import MultiHeadNN
from feature_selection import get_feature

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

    # Initialize lists to store the features and labels
    PDL_features = [0]*14
    ETL_features = [0]*14
    for j in range(14):
        Xf1_PDL, Xf1_ETL = get_feature(j,data_path= data_path)
        #print(len(Xf1_PDL),len(Xf1_PDL[0]))
        # Convert to numpy arrays if they aren't already
        Xf1_PDL = np.array(Xf1_PDL)  # Shape: (15, 600)
        Xf1_ETL = np.array(Xf1_ETL)  # Shape: (21, 600)
        #print(Xf1_ETL.shape)
        PDL_features[j]= Xf1_PDL
        ETL_features[j]= Xf1_ETL     
    PDL_features = np.array(PDL_features)  # Shape (14, 15, 600)
    ETL_features = np.array(ETL_features)  # Shape (14, 21, 600)

    # Reshape them into (14, 15*600) and (14, 21*600)
    PDL_reshaped = PDL_features.reshape(14, -1)  # Shape becomes (14, 15*600)
    ETL_reshaped = ETL_features.reshape(14, -1)  # Shape becomes (14, 21*600)


    #  Concatenate along axis 1 (the last axis)
    combined_features = np.concatenate((PDL_reshaped, ETL_reshaped), axis=1)  # Shape becomes (14, 15*600 + 21*600)

    # Step 3: Create labels (y) with 0 for PDL and 1 for ETL
    y_PDL = np.zeros(PDL_reshaped.shape[1], dtype=int)  # Labels 0 for PDL
    y_ETL = np.ones(ETL_reshaped.shape[1], dtype=int)   # Labels 1 for ETL

    # Concatenate the labels
    y = np.concatenate((y_PDL, y_ETL))  # Labels of length N (2*14 = 28 in this case)

    # Output shapes
    print("Combined features shape:", combined_features.shape)  # Shape (14, 15*600 + 21*600)
    print("y shape:", y.shape)  


    #  Convert numpy arrays to torch tensors
    features_tensor = torch.tensor(combined_features, dtype=torch.float32).T   # Convert features to tensor
    labels_tensor = torch.tensor(y, dtype=torch.long)  # Convert labels to tensor

    #  Create a TensorDataset
    dataset = TensorDataset(features_tensor, labels_tensor)

    # Create a DataLoader with an option for batching
    batch_size = 32  # Define your batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for x, y in dataloader:
        print(x.shape)
        print(y.shape)
        break

if __name__ == "__main__":
    main()