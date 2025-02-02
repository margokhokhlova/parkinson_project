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
import pdb


from feature_selection import get_feature
# from stack_features import stack_feature_lists, get_all_features_per_patient

from models import  train_model

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


class SimpleBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, num_layers=1):
        super(SimpleBiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 because of bidirectionality
        
    def forward(self, x):
        out, _ = self.bilstm(x)  # BiLSTM output
        out = out[:, -1, :]  # Take the last time step's output
        out = self.fc(out)  # Fully connected layer
        return out  # No sigmoid, use BCEWithLogitsLoss for numerical stability

# Custom dataset for gait data
class GaitDataset(Dataset):
    def __init__(self, PDL_features, ETL_features):
        if isinstance(PDL_features, torch.Tensor) and isinstance(ETL_features, torch.Tensor):
            # Convert to float32
            self.PDL_features = PDL_features.to(torch.float32)
            self.ETL_features = ETL_features.to(torch.float32)
        else:
            self.PDL_features = torch.tensor(PDL_features, dtype=torch.float32)  # Shape: (14, 14, 600) pat, feat, L
            self.ETL_features = torch.tensor(ETL_features, dtype=torch.float32)  # Shape: (14, 14, 600) pat, feat, L
        
        # Combine the two and create labels (0 for PDL, 1 for ETL)
        self.features = torch.cat((self.PDL_features, self.ETL_features), dim=0)  # Shape: (36, 14, 600)
        self.labels = torch.cat((
            torch.zeros(self.PDL_features.shape[0], dtype=torch.long),  # 0 for PDL
            torch.ones(self.ETL_features.shape[0], dtype=torch.long)    # 1 for ETL
        ))
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Transpose to (sequence_length, num_features), i.e., (600, 14)
        return self.features[idx].transpose(0, 1), self.labels[idx]  
    



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


    new_path=save_path+f'/vanilla_bilstm'  
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    current_feat = [0,1,2,3,4,5,6,8,9,12,13]
    num_feat = len(current_feat)
    # Initialize lists to store the features and labels
    PDL_features = [0]*num_feat
    ETL_features = [0]*num_feat    

    #     All but left, right, timeSignal.    
    for i, j in enumerate(current_feat):
        Xf1_PDL, Xf1_ETL = get_feature(j,data_path= data_path,  min_len_established = 600, skip_patients = [55, 70,71,72,73,74,75,76])
        Xf1_PDL = np.array(Xf1_PDL)  # Shape: (14, 600)
        Xf1_ETL = np.array(Xf1_ETL)  # Shape: (14, 600)
        #print(Xf1_ETL.shape)
        PDL_features[i]= Xf1_PDL
        ETL_features[i]= Xf1_ETL         
    
    PDL_features = np.array(PDL_features)  # Shape (14, 14, 600)
    PDL_features = np.transpose(PDL_features, (1, 0, 2))  # Shape (14, 14, 600)


    # Reshape ETL_features to (21, 14, 600)
    ETL_features = np.array(ETL_features)  # Shape (14, 21, 600)
    ETL_features = np.transpose(ETL_features, (1, 0, 2))  # Shape (21, 14, 600)

    assert(PDL_features.shape == ETL_features.shape)




    accuracies= []
    # repeat N times
    for iter in range(100):
        
        # divide into train and test features
        # Randomly shuffle indices and split into train and test
        indices = np.random.permutation(PDL_features.shape[0])  # Randomly shuffle indices for the first dimension
        test_indices = indices[:7]  # First 7 indices for the test set
        train_indices = indices[7:]  # Remaining indices for the training set


        #min max normalization for the features for TRAIN
        all_train = np.concatenate((PDL_features[train_indices], ETL_features[train_indices]), axis=0) 
        all_train =torch.tensor(all_train) #N train, (14 samples, 11 features, 600 length)
        # Step 1: Reshape to combine patients and time dimensions for feature normalization
        reshaped_features = all_train.permute(1, 0, 2).reshape(num_feat, -1)  # Shape: (features, N train *600)

        # Step 2: Compute min and max values for each feature
        min_values = reshaped_features.min(dim=1, keepdim=True).values  # Shape: (features, 1)
        max_values = reshaped_features.max(dim=1, keepdim=True).values  # Shape: (features, 1)

        normalized_features_train  = (reshaped_features - min_values) / (max_values - min_values + 1e-8)  # Avoid division by zero
        normalized_features_train = normalized_features_train.reshape(num_feat, len(train_indices)*2, 600) # back to features, N train samples, length
        normalized_features_train  = normalized_features_train.permute(1, 0, 2) # reshape back swipping Features & N => N, F, L

        assert np.allclose(all_train, reshaped_features.reshape(num_feat, len(train_indices)*2, 600).permute(1, 0, 2)), "Arrays are not close enough"

        PDL_train = normalized_features_train[:len(train_indices),:,:]
        ETL_train =  normalized_features_train[len(train_indices):,:,:]

        #min max normalization for the features for TEST
        all_test = np.concatenate((PDL_features[test_indices], ETL_features[test_indices]), axis=0) 
        all_test =torch.tensor(all_test)
        reshaped_features_test = all_test.permute(1, 0, 2).reshape(num_feat, -1)  # Shape: (14, 14*600)
        normalized_features_test = (reshaped_features_test - min_values) / (max_values - min_values + 1e-8)  # Avoid division by zero
        normalized_features_test = normalized_features_test.reshape(num_feat, len(test_indices)*2, 600) 
        normalized_features_test  = normalized_features_test.permute(1, 0, 2) # reshape back

        PDL_test = normalized_features_test[:len(test_indices),:,:]
        ETL_test =  normalized_features_test[len(test_indices):,:,:]


        # Create dataset and dataloader
        gait_dataset_train = GaitDataset(PDL_train, ETL_train)
        dataloader = DataLoader(gait_dataset_train, batch_size=4, shuffle=True)

        # Model parameters
        input_size = 11 # Number of features
        hidden_size = 4  # Hidden size 
        output_size = 1   # Binary classification (PDL or ETL)
        num_epochs = 120

        model = SimpleBiLSTM(input_size, hidden_size, output_size)


            # Training setup
        criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        # Training loop

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        model.to(device)

        model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Track the total loss
                total_loss += loss.item() * inputs.size(0)
                
                # Convert probabilities to binary predictions (0 or 1)
                predicted = (outputs.squeeze() >= 0.5).long()
                
                # Track the number of correct predictions
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
            
            # Calculate average loss and accuracy
            avg_loss = total_loss / total_samples
            accuracy = correct_predictions / total_samples * 100
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        print("Training complete.")
        # TEST evaluation
        gait_dataset_test = GaitDataset(PDL_test, ETL_test)
        test_dataloader = DataLoader(gait_dataset_test, batch_size=4, shuffle=False)


        model.eval()  # Set the model to evaluation mode
        test_correct_predictions = 0
        test_total_samples = 0

        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Convert probabilities to binary predictions (0 or 1)
                predicted = (outputs.squeeze() >= 0.5).long()
                
                # Track the number of correct predictions
                test_correct_predictions += (predicted == labels).sum().item()
                test_total_samples += labels.size(0)

        # Calculate test accuracy
        test_accuracy = test_correct_predictions / test_total_samples * 100
        print(f"Test Accuracy: {test_accuracy:.2f}% on {test_total_samples} test samples")
        accuracies.append(test_accuracy)
    print(f'Mean accuracy is {np.mean(accuracies)}, all accuracies {accuracies}')

if __name__ == "__main__":
    main()
