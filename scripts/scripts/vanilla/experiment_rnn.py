import yaml
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



from feature_selection import get_feature
from stack_features import stack_feature_lists, get_all_features_per_patient
import viz
from models import SimpleNN, train_model

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


    new_path=save_path+f'/all_features'  
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
    X_tensor, y_tensor = stack_feature_lists(PDL_features, ETL_features)
    print(X_tensor.shape)  # Expected output: torch.Size([21600, 7])
    print(y_tensor.shape)  # Expected output: torch.Size([43200])
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # model training and all that
    model = SimpleNN(14,16) #just 2 features
    train_model(model, train_loader, num_epochs=num_epochs,path = new_path+f'/all_features.pth')
    #model = torch.load(new_path+f'/all_features.pth', weights_only=False)
    # check trained model on thestate_dic same features from clasification
    X_PDL = get_all_features_per_patient(PDL_features)    
     # Store results for each patient
    patient_like_counts = []
    n,_,l = X_PDL.shape    
    # Store results for visualization
    pd_counts = []
    et_counts = []
    print(X_PDL.shape)
    for i in range(n):
        pd_like_count = 0
        et_like_count = 0
        patient_data = X_PDL[i,:,:]
        for j in range(l):
            sample = patient_data[:,j]
            sample_tensor = torch.from_numpy(sample).float().unsqueeze(0)
            with torch.no_grad():
                prediction = model(sample_tensor)

            if prediction.item() > 0.5:
                pd_like_count += 1
            else:
                et_like_count += 1

        pd_counts.append(pd_like_count)
        et_counts.append(et_like_count)

        patient_like_counts.append((pd_like_count, et_like_count))
        print(f"Patient {n}: PDL")
        print(f"  PD-like samples: {pd_like_count}")
        print(f"  ET-like samples: {et_like_count}")
        print("\n")
    # Plotting
    patients = range(1, n + 1)

    plt.figure(figsize=(10, 6))
    plt.bar(patients, pd_counts, color='red', label='PD-like')
    plt.bar(patients, et_counts, color='blue', bottom=pd_counts, label='ET-like')
    plt.xlabel('Patient Number')
    plt.ylabel('Number of Samples')
    plt.title(f'PD-like vs ET-like Sample Counts per Patient for PDL patients')
    plt.legend()
    plt.xticks(patients)
    plt.savefig(new_path+f'/PDL.png')
    # Clear the current figure to free up memory

    # Do the same for ETL
    X_ETL = get_all_features_per_patient(ETL_features)    
     # Store results for each patient
    patient_like_counts = []
    n,_,l =  X_ETL.shape    
    # Store results for visualization
    pd_counts = []
    et_counts = []
    for i in range(n):
        pd_like_count = 0
        et_like_count = 0
        patient_data = X_ETL[i,:,:]
        for j in range(l):
            sample = patient_data[:,j]
            sample_tensor = torch.from_numpy(sample).float().unsqueeze(0)
            with torch.no_grad():
                prediction = model(sample_tensor)

            if prediction.item() > 0.5:
                pd_like_count += 1
            else:
                et_like_count += 1

        pd_counts.append(pd_like_count)
        et_counts.append(et_like_count)

        patient_like_counts.append((pd_like_count, et_like_count))
        print(f"Patient {n}: ETL")
        print(f"  PD-like samples: {pd_like_count}")
        print(f"  ET-like samples: {et_like_count}")
        print("\n")
    # Plotting
    patients = range(1, n + 1)

    plt.figure(figsize=(10, 6))
    plt.bar(patients, pd_counts, color='red', label='PD-like')
    plt.bar(patients, et_counts, color='blue', bottom=pd_counts, label='ET-like')
    plt.xlabel('Patient Number')
    plt.ylabel('Number of Samples')
    plt.title(f'PD-like vs ET-like Sample Counts per Patient for ETL patients')
    plt.legend()
    plt.xticks(patients)
    plt.savefig(new_path+f'/ETL.png')
    # Clear the current figure to free up memory

   
if __name__ == "__main__":
    main()