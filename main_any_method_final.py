import numpy as np
from data_outils import dataloader
from data_outils.data_features import *
from data_outils.data_perm import *
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
def train_model(X_train, y_train, X_val, y_val, num_epochs=50, lr=0.001):
    model = SimpleNN(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = SimpleNN(X_train.shape[1])
                best_model.load_state_dict(model.state_dict())

    return best_model

def evaluate(model, X_test, y_test):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = (outputs >= 0.5).float()
        accuracy = (preds == y_test_tensor).float().mean().item()
    return accuracy

#all_combinations = [range(14),[8,11,13], [1,8,11],[3,6,11],[7,8,10],[7,10,11,13],[6,0,4,3,1,5,8,11,13],[6,0,4,3,1,5,8],[6,8], [0], [1], [2],[6]]
all_combinations = [[0]]
def main():
    for model_features in all_combinations : 
        # to log all the experiments and load data, to change
        sys.stdout = open('/home/margokat/Documents/projects/margo_files/rocket/results/result_rocket_val.txt', 'a')
        data_path = '/home/margokat/Documents/projects/margo_files/data/data_lstm_august24_PDET_left_right.csv'
        # parameters
        min_len_established = 600
        num_val_cycles = 100
        num_feat = len(model_features)
        print('model_features:',model_features)
        print('num_feat:',num_feat)
        print('selected_features:',[feat_order[j] for j in model_features if j <= 15])
        # Initialize lists to store the features and labels
        PDL_features = [0]*num_feat
        ETL_features = [0]*num_feat	
        for i, j in enumerate(model_features):
            #print('j:',j)
            Xf1_PDL, Xf1_ETL = get_feature(
                j,
                data_path= data_path,
                min_len_established= min_len_established,
                skip_patients= [55, 70,71,72,73,74,75,76])

            Xf1_PDL = np.array(Xf1_PDL)  # Shape: (14, 600)
            Xf1_ETL = np.array(Xf1_ETL)  # Shape: (14, 600)

            PDL_features[i]= Xf1_PDL
            ETL_features[i]= Xf1_ETL	

        #
        PDL_features = np.array(PDL_features)  # Shape (14, 14, 600)
        PDL_features = np.transpose(PDL_features, (1, 0, 2))  # Shape (14, 14, 600)  if all patients, see skip
        ETL_features = np.array(ETL_features)  # Shape (14, 21, 600)
        ETL_features = np.transpose(ETL_features, (1, 0, 2))  # Shape (21, 14, 600) if all patients, see skip
        assert(PDL_features.shape == ETL_features.shape)
        
        optimal_kernel = 1_0000 
        # make iterations over all set with one pair leave out  (14 × 14 = 196 total combinations).
        # Loop over leave-one-out combinations

        val_accuracy_array = []
        test_accuracy_array = []

        for X_trainval, y_trainval, X_test, y_test, (i, j) in leave_one_pair_out(PDL_features, ETL_features):
            print(f"Left out PDL[{i}] and ETL[{j}]")
            
            models = []
            val_accuracies = []

            for _ in range(num_val_cycles):
                # Train-validation split
                X_train, y_train, X_val, y_val = stratified_random_split(X_trainval, y_trainval, val_size=4)

                # Normalize using training split only
                normalizer = FeatNormalisation(num_feat=len(model_features))
                normalizer.calculate_norm_values(X_train)
                
                X_train = normalizer.normalize(X_train).reshape(X_train.shape[0], -1)
                X_val = normalizer.normalize(X_val).reshape(X_val.shape[0], -1)
                X_test_norm = normalizer.normalize(X_test).reshape(X_test.shape[0], -1)

                # Train model
                model = train_model(X_train, y_train, X_val, y_val)
                val_acc = evaluate(model, X_val, y_val)

                models.append((model, X_test_norm))  # Save test set version for this fold
                val_accuracies.append(val_acc)

            # Choose best model (highest val acc)
            best_idx = np.argmax(val_accuracies)
            best_model, best_X_test = models[best_idx]

            # Evaluate once on the test pair
            test_acc = evaluate(best_model, best_X_test, y_test)

            print(f"Best Validation Accuracy: {val_accuracies[best_idx]:.4f}, Test Accuracy: {test_acc:.4f} for combination {model_features}")

            val_accuracy_array.append(val_accuracies[best_idx])
            test_accuracy_array.append(test_acc)
        # evaluate then the mean and std for all pairs, test and val
        val_mean = np.mean(val_accuracy_array)
        val_std = np.std(val_accuracy_array)
        test_mean = np.mean(test_accuracy_array)
        test_std = np.std(test_accuracy_array)

        print("\n=== Final Summary ===")
        print(f"Validation Accuracy: Mean = {val_mean:.4f}, Std = {val_std:.4f}")
        print(f"Test Accuracy:       Mean = {test_mean:.4f}, Std = {test_std:.4f}")


if __name__ == "__main__":
	main()