import yaml
import os
import time
import random
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from models import MultiHeadPairsNN, train_model
from stack_features import stack_features

import viz
from feature_selection import get_feature
from torch.utils.data import Dataset, DataLoader

    
class MultiHeadDataset(Dataset):
    def __init__(self, x, y):
        """
        Args:
            x: Tensor of shape [N, 2, 91], input features.
            y: Tensor of shape [N], target labels.
        """
        self.x = x
        self.y = y

    def __len__(self):
        # Number of samples
        return self.x.size(0)

    def __getitem__(self, idx):
        """
        Args:
            idx: Index of the sample to fetch.
        
        Returns:
            A tuple of (features, target).
        """
        return self.x[idx], self.y[idx]


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
    save_path = config['settings']['save_path']
    feature_names =  config['settings']['feat_order']
    combinations = config['settings']['possible_pairs']
    
    # Print out the configuration
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Model Name: {model_name}")
    print(f"Data Path: {data_path}")
    print(f"Feature_names: {feature_names}")


    train_ETL =  list(range(41, 55))
    train_PDL = list(range(56, 70))
    accuracies= []
    # repeat N times
    for iter in range(100):
        start_time = time.time()
        # patients we decided to exclude
        skip_patients = [55, 70,71,72,73,74,75,76]
        test_patients = random.sample(train_ETL, 7) + random.sample(train_PDL, 7)
        skip_patients+=test_patients
        print(len(skip_patients), skip_patients)
        # get through all the pairs.
        X_multihead_list = []  # A list to collect all tensors
        for pair in combinations:    
            Xf1_PDL, Xf1_ETL = get_feature(pair[0],data_path= data_path,   min_len_established = 600, skip_patients=skip_patients)
            Xf2_PDL, Xf2_ETL = get_feature(pair[1],data_path= data_path,  min_len_established = 600, skip_patients=skip_patients)
            X_tensor, y_tensor = stack_features(Xf1_PDL,Xf2_PDL, Xf1_ETL,Xf2_ETL)
            # print(y_tensor.shape)
            X_multihead_list.append(X_tensor.unsqueeze(-1))  # Add a new dimension at the end
    
        # Stack all tensors along the last dimension to get shape [16800, 2, 91]
        X_multihead = torch.cat(X_multihead_list, dim=-1)
        print(X_multihead.shape)  # Should output: torch.Size([16800, 2, 91])
        print(y_tensor.shape)
        dataset = MultiHeadDataset(X_multihead, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch_x, batch_y in dataloader:
            print("Batch X shape:", batch_x.shape, "type: ", type(batch_x))  # Expected: [batch_size, 2, 91]
            print("Batch Y shape:", batch_y.shape)  # Expected: [batch_size]
            break
        
        model=MultiHeadPairsNN(input_dim=2, hidden_dim=1, num_heads=91, num_layers=1, merge_opt = 'weighted')
        print('Training the model') 
        train_model(model, dataloader, num_epochs=num_epochs, path=None)

        # getting training quantities of wavetrains
        dataloader = DataLoader(dataset, batch_size=600, shuffle=False) # normally, we should have consequent data this way
        train_PD = []
        train_y  = []
        for batch_x, batch_y in dataloader:
            assert torch.unique(batch_y).numel() == 1, "Tensor does not have a single unique value."
            predicted_wavetrain_labels = model(batch_x)
            num_class_1 = (predicted_wavetrain_labels > 0.5).sum().item()
            train_PD.append([num_class_1, 600-num_class_1])
            train_y.append(batch_y[0])
        classifier = LogisticRegression()
        print(train_PD, train_y)
        classifier.fit(train_PD, train_y)

        # testing of the model
        #TODO rewrite so that dataloader has all the patients, and then batch size is 600, and the order is correct. 
        correct_pred = 0
        for patient in test_patients:
            X_multihead_list = []  # A list to collect all tensors
            all_patients =  train_ETL + train_PDL + [55, 70,71,72,73,74,75,76]
            all_patients.remove(patient) 
            # print(patient,  all_patients)       
            for pair in combinations:    
                Xf1_PDL, Xf1_ETL = get_feature(pair[0],data_path= data_path,   min_len_established = 600, skip_patients= all_patients)
                Xf2_PDL, Xf2_ETL = get_feature(pair[1],data_path= data_path,  min_len_established = 600, skip_patients= all_patients)
                #print("each time just two should be non-zero: ", len(Xf1_PDL), len(Xf1_ETL), len(Xf2_PDL), len(Xf2_ETL)) 
                X_tensor, y_tensor = stack_features(Xf1_PDL,Xf2_PDL, Xf1_ETL,Xf2_ETL)
                X_multihead_list.append(X_tensor.unsqueeze(-1))  # Add a new dimension at the end
            # predict the labels for the patient's wavetrains 
            X_multihead = torch.cat(X_multihead_list, dim=-1)
            #print(X_multihead.shape, type(X_multihead))
            test_dataset = MultiHeadDataset(X_multihead, y_tensor)
            dataloader = DataLoader(test_dataset, batch_size=600, shuffle=False)
            for batch_x, batch_y in dataloader:
                predicted_wavetrain_labels = model(batch_x)
                num_class_1 = (predicted_wavetrain_labels > 0.5).sum().item()
                print(f'Patient {patient} predicted PD  wavetrains {num_class_1}')  
            sample = np.array([num_class_1, 600-num_class_1]) 
            y_pred = classifier.predict(sample.reshape(1, -1)) 
            if y_pred == batch_y[0]:
                correct_pred+=1
        print(f"Final accuracy is {iter} {correct_pred/len(test_patients)}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Random data selection interation time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} secondes")
        accuracies.append(correct_pred/len(test_patients))
    print(f'Mean accuracy is {np.mean(accuracies)}, all accuracies {accuracies}')
if __name__ == "__main__":
    main()
