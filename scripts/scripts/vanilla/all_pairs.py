import yaml
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from feature_selection import get_feature
from stack_features import stack_features, get_min_max_values, get_per_patient_features
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


    best_model_index_accuracy = {}
    for pair in combinations:          
        new_path=save_path+f'/{pair[0]}_{pair[1]}'       
        print(f'Starting to train for the first pair {pair[0]}_{pair[1]}, results are saved {new_path}')
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        Xf1_PDL, Xf1_ETL = get_feature(pair[0],data_path= data_path)
        Xf2_PDL, Xf2_ETL = get_feature(pair[1],data_path= data_path)
        X_tensor, y_tensor = stack_features(Xf1_PDL,Xf2_PDL, Xf1_ETL,Xf2_ETL)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        viz.save_feature_scatter_plot(train_loader, new_path+'/scatter_plot.png',feature_names[pair[0]], feature_names[pair[1]])
        # model training and all that
        model = SimpleNN(2,16) #just 2 features
        train_model(model, train_loader, num_epochs=num_epochs,path = new_path+f'/{feature_names[pair[0]]}_{feature_names[pair[1]]}.pth')
        # plotting and training results
        min_f1, max_f1, min_f2, max_f2 = get_min_max_values(Xf1_PDL,Xf2_PDL, Xf1_ETL,Xf2_ETL)
        xx, yy = np.meshgrid(np.linspace(min_f1, max_f1, 100), np.linspace(min_f2, max_f2, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = torch.tensor(grid, dtype=torch.float32) 
        # Get the network outputs for the grid and plot the output
        with torch.no_grad():
            outputs = model(grid_tensor).numpy()   
        viz.plot_probabilistic_output(outputs, new_path+f'/proba_{feature_names[pair[0]]}_{feature_names[pair[1]]}')
        # finally, plot the resulting histograms per number of features
        X_PDL, X_ETL = get_per_patient_features(Xf1_PDL,Xf2_PDL, Xf1_ETL,Xf2_ETL)
        pdl_features = viz.viz_quantity_features(X_PDL, 'PDL', model, new_path+f'/PDL_quant_{feature_names[pair[0]]}_{feature_names[pair[1]]}')
        etl_features = viz.viz_quantity_features(X_ETL, 'ETL', model, new_path+f'/ETL_quant_{feature_names[pair[0]]}_{feature_names[pair[1]]}')
        # try to classify them
        X_quant = np.vstack((pdl_features,  etl_features))
        # Create labels: 1 for PD-like, 0 for ET-like
        y_pdl = np.ones(pdl_features.shape[0])  # Labels for PD-like patients
        y_etl = np.zeros(etl_features.shape[0])  # Labels for ET-like patients

        # Concatenate the labels
        y_quant = np.concatenate((y_pdl, y_etl))

        # Split the data into training and test sets
        # Store the accuracy scores for each run
        accuracies = []

        # Run the experiment 50 times on different train and test sets
        for _ in range(50):
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_quant, y_quant, test_size=0.2)

            # Initialize and train the classifier (Logistic Regression in this example)
            classifier = LogisticRegression()
            classifier.fit(X_train, y_train)

            # Predict on the test set
            y_pred = classifier.predict(X_test)

            # Evaluate the classifier
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # Calculate the mean and standard deviation of the accuracies
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        # Report the results
        print(f"{feature_names[pair[0]]}_{feature_names[pair[1]]} Mean classification accuracy: {mean_accuracy:.2f}")
        print(f"Standard deviation of accuracy: {std_accuracy:.2f}")
    best_model_index_accuracy[f'{feature_names[pair[0]]}_{feature_names[pair[1]]}'] = mean_accuracy

if __name__ == "__main__":
    main()