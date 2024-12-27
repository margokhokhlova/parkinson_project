import yaml
import os
import torch
import sys
print(sys.path)
sys.path.append('.') # 2024-10-20, A.M.
sys.path.append('/home/mkhokhlo/projects/kotelnikov/scripts/scripts/vanilla') #TODO - clean it
sys.path.append('/home/mkhokhlo/projects/kotelnikov/scripts/scripts/') #TODO - clean it 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models import SimpleBinaryClassifier, train_model, EnhancedBinaryClassifier
import viz
from feature_selection import get_feature

def load_config(config_file):
	with open(config_file, 'r') as file:
		config = yaml.safe_load(file)
	return config
def main():
	# Load the configuration from the YAML file
	# 2024-10-20, A.M.
	# config = load_config('scripts/scripts/config.yaml')
	config = load_config('/home/mkhokhlo/projects/kotelnikov/scripts/scripts/config.yaml')

	# Access the settings from the config
	learning_rate = config['settings']['learning_rate']
	batch_size = config['settings']['batch_size']
	num_epochs = config['settings']['num_epochs']
	model_name = config['settings']['model_name']
	data_path = config['settings']['data_path']
	save_path = config['settings']['save_path']
	feature_names =  config['settings']['feat_order']
	integration_mode = config['settings']['integration_mode']
	save_path= save_path + "_all_14"
	#
	# Print out the configuration
	print(f"Config: {config}")

	# Initialize lists to store the features and labels
	initial_PDL_features = [0]*14
	initial_ETL_features = [0]*14
	for j in range(14):
		Xf1_PDL,Xf1_ETL= get_feature(j,data_path= data_path)
		#print(len(Xf1_PDL),len(Xf1_PDL[0]))
		# Convert to numpy arrays if they aren't already
		Xf1_PDL= np.array(Xf1_PDL)  # Shape: (15, 600)
		Xf1_ETL= np.array(Xf1_ETL)  # Shape: (21, 600)
		#print(Xf1_ETL.shape)
		initial_PDL_features[j]= Xf1_PDL
		initial_ETL_features[j]= Xf1_ETL
	initial_PDL_features = np.array(initial_PDL_features)  # Shape (14, 15, 600)
	initial_ETL_features = np.array(initial_ETL_features)  # Shape (14, 21, 600)
	#print("(1)initial_PDL_features.shape:",initial_PDL_features.shape)
	#print("(1)initial_ETL_features.shape:",initial_ETL_features.shape)
	initial_PDL_features= initial_PDL_features.transpose(1,0,2)
	initial_ETL_features= initial_ETL_features.transpose(1,0,2)
	#print("(2)initial_PDL_features.shape:",initial_PDL_features.shape)
	#print("(2)initial_ETL_features.shape:",initial_ETL_features.shape)

	accuracies= []
	model_train_accuracy = []

	# Run the experiment 1000 times on different train and test sets
	for experiment in range(30):
		print("***** ",experiment," *****")
		#
		################################################################
		# TRAIN                                                        #
		################################################################
		#
		# Split the dataset into training and testing sets
		X_train_PDL, X_test_PDL= train_test_split(initial_PDL_features,test_size=0.3)
		X_train_ETL, X_test_ETL= train_test_split(initial_ETL_features,test_size=0.2)
		print("Test PDL_features.shape:",X_test_PDL.shape, "Test ETL_features.shape:",X_test_ETL.shape)
		# Reshape them into (14, 15*600) and (14, 21*600)
		PDL_reshaped = X_train_PDL.reshape(14, -1)  # Shape becomes (14, 15*600)
		ETL_reshaped = X_train_ETL.reshape(14, -1)  # Shape becomes (14, 21*600)
		#  Concatenate along axis 1 (the last axis)
		combined_features = np.concatenate((PDL_reshaped, ETL_reshaped), axis=1)  # Shape becomes (14, 15*600 + 21*600)
		# experiment 3: Create labels (y) with 0 for PDL and 1 for ETL
		y = np.concatenate((np.ones(PDL_reshaped.shape[1], dtype=int),np.zeros(ETL_reshaped.shape[1], dtype=int)))  # Labels for training
		#  Convert numpy arrays to torch tensors
		features_tensor = torch.tensor(combined_features, dtype=torch.float32).T   # Convert features to tensor
		labels_tensor = torch.tensor(y, dtype=torch.float)  # Convert labels to tensor
		#  Create a TensorDataset
		dataset = TensorDataset(features_tensor, labels_tensor)
		# Create a DataLoader with an option for batching
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
		model= EnhancedBinaryClassifier(input_dim=14)
		train_acc = train_model(model, dataloader, num_epochs=num_epochs, learning_rate=learning_rate, path=save_path+'/vanilla_multi_14.pth')
		train_acc = 0
		# model= torch.load(save_path+'/vanilla_multi_14.pth', weights_only=False)
		# now let's viz the results, and train simple classfier 
		model_train_accuracy.append(train_acc)
		PDL_features= np.swapaxes(X_train_PDL, 1,2)
		ETL_features= np.swapaxes(X_train_ETL, 1,2)
		pdl_classif= viz.viz_quantity_features(PDL_features, 'PDL', model,save_path+'/PDL_quant_train_14_')
		etl_classif= viz.viz_quantity_features(ETL_features, 'ETL', model,save_path+'/ETL_quant_train_14_')
		#print("pdl_classif:",pdl_classif)
		#print("etl_classif:",etl_classif)
		# try to classify them
		X_quant= np.vstack((pdl_classif,etl_classif))
		y_pdl = np.ones(pdl_classif.shape[0])  # Labels 1 for pdl_classif
		y_etl = np.zeros(etl_classif.shape[0])  # Labels 0 for etl_classif
		y = np.concatenate((y_pdl, y_etl))
		classifier= LogisticRegression()
		classifier.fit(X_quant,y)
		#
		################################################################
		# TEST                                                         #
		################################################################
		#
		PDL_features= X_test_PDL
		ETL_features= X_test_ETL
		# Reshape them into (14, 15*600) and (14, 21*600)
		PDL_reshaped= PDL_features.reshape(14,-1)  # Shape becomes (14, 15*600)
		ETL_reshaped= ETL_features.reshape(14,-1)  # Shape becomes (14, 21*600)
		# Concatenate along axis 1 (the last axis)
		# experiment 3: Create labels (y) with 0 for PDL and 1 for ETL
		y_PDL= np.ones(PDL_reshaped.shape[1],dtype=int) # Labels 1 for PDL
		y_ETL= np.zeros(ETL_reshaped.shape[1],dtype=int) # Labels 0 for ETL
		# Concatenate the labels
		y = np.concatenate((y_PDL,y_ETL))  # Labels for test
		# do test and viz
		PDL_features= np.swapaxes(PDL_features, 1,2)
		ETL_features= np.swapaxes(ETL_features,1,2)
		pdl_classif= viz.viz_quantity_features(PDL_features, 'PDL', model,save_path+'/PDL_quant_test_14_')
		etl_classif= viz.viz_quantity_features(ETL_features, 'ETL', model,save_path+'/ETL_quant_test_14_')
		# try to classify them
		y_pdl = np.ones(pdl_classif.shape[0])  # Labels 1 for pdl_classif
		y_etl = np.zeros(etl_classif.shape[0])  # Labels 0 for etl_classif
		y = np.concatenate((y_pdl, y_etl))
		X_quant= np.vstack((pdl_classif,etl_classif))
		y_pred= classifier.predict(X_quant)
		# Evaluate the classifier
		accuracy= accuracy_score(y,y_pred)
		print(f"Predicted {y_pred}, test GT labels {y}")
		accuracies.append(accuracy)
		
	# Calculate the mean and standard deviation of the accuracies in all the model runs
	mean_accuracy= np.mean(accuracies)
	median_accuracy= np.median(accuracies)
	std_accuracy= np.std(accuracies)
	min_accuracy= np.min(accuracies)
	max_accuracy= np.max(accuracies)
	print(f"MEDIAN classification accuracy: {median_accuracy:.3f}")
	print(f"MEAN classification accuracy: {mean_accuracy:.3f}")
	print(f"Minimal classification accuracy: {min_accuracy:.3f}")
	print(f"Maximal classification accuracy: {max_accuracy:.3f}")
	print(f"Standard deviation of accuracy: {std_accuracy:.3f}")
	print(f"Model train accuracy {np.mean(model_train_accuracy)}")
if __name__ == "__main__":
	main()
