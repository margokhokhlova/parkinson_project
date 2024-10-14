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



from feature_selection import get_feature
from stack_features import stack_feature_lists, get_all_features_per_patient

from models import SimpleNN, train_model

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Apply sigmoid for binary classification
        self.hidden_size = hidden_size
        
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        out, _ = self.rnn(x, h_0)  # Pass through RNN layer
        out = out[:, -1, :]  # Get the last time step's output
        out = self.fc(out)   # Pass through the final fully connected layer
        out = self.sigmoid(out)  # Apply sigmoid for binary classification
        return out

# Custom dataset for gait data
class GaitDataset(Dataset):
    def __init__(self, PDL_features, ETL_features):
        self.PDL_features = torch.tensor(PDL_features, dtype=torch.float32)  # Shape: (15, 14, 600)
        self.ETL_features = torch.tensor(ETL_features, dtype=torch.float32)  # Shape: (21, 14, 600)
        
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


    new_path=save_path+f'/vanilla_rnn'  
    if not os.path.exists(new_path):
        os.makedirs(new_path)


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
    PDL_features = np.transpose(PDL_features, (1, 0, 2))  # Shape (15, 14, 600)

    # Reshape ETL_features to (21, 14, 600)
    ETL_features = np.array(ETL_features)  # Shape (14, 21, 600)
    ETL_features = np.transpose(ETL_features, (1, 0, 2))  # Shape (21, 14, 600)
        

    # Create dataset and dataloader
    gait_dataset = GaitDataset(PDL_features, ETL_features)
    dataloader = DataLoader(gait_dataset, batch_size=4, shuffle=True)

    # Model parameters
    input_size = 14  # Number of features
    hidden_size = 32  # Hidden size for RNN
    output_size = 1   # Binary classification (PDL or ETL)

    model = SimpleRNN(input_size, hidden_size, output_size)


        # Training setup
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
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

if __name__ == "__main__":
    main()