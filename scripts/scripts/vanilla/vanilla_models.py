import yaml
import os
import torch
import sys
import inspect
from sklearn.metrics import roc_curve


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models import SimpleBinaryClassifier, train_model
import viz
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
    y_PDL = np.ones(PDL_reshaped.shape[1], dtype=int)  # Labels 0 for PDL
    y_ETL = np.zeros(ETL_reshaped.shape[1], dtype=int)   # Labels 1 for ETL

    # Concatenate the labels
    y = np.concatenate((y_PDL, y_ETL))  # Labels of length N (2*14 = 28 in this case)

    # Output shapes
    print("Combined features shape:", combined_features.shape)  # Shape (14, 15*600 + 21*600)
    print("y shape:", y.shape)  


    #  Convert numpy arrays to torch tensors
    features_tensor = torch.tensor(combined_features, dtype=torch.float32).T   # Convert features to tensor
    labels_tensor = torch.tensor(y, dtype=torch.float)  # Convert labels to tensor


    #  Create a TensorDataset
    dataset = TensorDataset(features_tensor, labels_tensor)

    # Create a DataLoader with an option for batching
    batch_size = batch_size  # Define your batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # now let's prepare features to run the tests
    PDL_features =  np.swapaxes(np.swapaxes(PDL_features, 0, 1), 1,2)
    ETL_features =  np.swapaxes(np.swapaxes(ETL_features, 0, 1),1,2)
    youden_j_stack = [0,1,2,3,4,5,10,50,100]
    for i, num_epochs in enumerate([0, 1,2,3,4,5,10,50,100]):
        model=SimpleBinaryClassifier(input_dim=14)
        train_model(model, dataloader, num_epochs=num_epochs, path=save_path+'/vanilla_14.pth')
        if i!=0:
            model = torch.load(save_path+'/vanilla_14.pth', weights_only=False)
        else:
            print('\n No model, check again')
        pdl_classif = viz.viz_quantity_features(PDL_features, 'PDL', model,save_path+'/PDL_quant_14_')
        etl_classif = viz.viz_quantity_features(ETL_features, 'ETL', model,save_path+'/ETL_quant_14_')

        # make classification using Youden's  and histograms
        # Example arrays, using the first column for classification
        pdl_classif = pdl_classif[:, 0]  # Get first column
        etl_classif = etl_classif[:, 0]  # Get first column

        # Concatenate data and create labels
        all_histograms = np.concatenate((pdl_classif, etl_classif))
        print(all_histograms)
        all_labels = np.array([1]*len(pdl_classif) + [0]*len(etl_classif))

        # Calculate ROC curve to find sensitivity, specificity, and thresholds
        fpr, tpr, thresholds = roc_curve(all_labels, all_histograms)

        # Calculate Youden's J statistic
        youden_j = tpr - fpr  # Sensitivity + Specificity - 1

        # Find the index with the maximum Youden's J
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]

        print(f"Optimal Threshold: {optimal_threshold}")
        print(f"Maximum Youden's J statistic: {youden_j[optimal_idx]}")


        viz.plot_joudens(pdl_classif, etl_classif, optimal_threshold, thresholds, youden_j, optimal_idx, save_path+f'/Jouden_threshold_{num_epochs}')
        youden_j_stack[i]= youden_j[optimal_idx]
    print(f'Epochs: [0,1,2,3,4,5,10,50,100] {youden_j_stack}')
if __name__ == "__main__":
    main()