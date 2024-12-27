import yaml
import os
import torch
import sys
print(sys.path)
sys.path.append('.') # 2024-10-20, A.M.
# sys.path.append('/home/mkhokhlo/projects/kotelnikov/scripts/scripts') #TODO - clean it

import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models import SimpleBinaryClassifier, train_model
from models import Alexey_net_Step, LeftRightLinear, Alexey_net_Rect_Vec
import viz
from feature_selection import get_feature
from create_data_008a import CreateData

def load_config(config_file):
	with open(config_file, 'r') as file:
		config = yaml.safe_load(file)
	return config
def init_weights(m):
	if isinstance(m,LeftRightLinear):
		# print("LeftRightLinear!!!")
		# torch.nn.init.constant_(m.weight,-1.0)
		# torch.nn.init.constant_(m.weight,-1.0)
		if m.leftBias is not None:
			# torch.nn.init.constant_(m.leftBias,-1000.0)
			# torch.nn.init.constant_(m.leftBias,450.0)
			# torch.nn.init.constant_(m.leftBias,-450.0)
			torch.nn.init.constant_(m.leftBias,0.0)
		if m.rightBias is not None:
			# torch.nn.init.constant_(m.rightBias,1000.0)
			# torch.nn.init.constant_(m.rightBias,450.0)
			# torch.nn.init.constant_(m.rightBias,-450.0)
			torch.nn.init.constant_(m.rightBias,0.0)
	else:
		if isinstance(m,torch.nn.Linear):
			print("Linear!!!")
			torch.nn.init.constant_(m.weight,0.5)
			# torch.nn.init.constant_(m.weight,-1.0)
			if m.bias is not None:
				torch.nn.init.constant_(m.bias,0)
				# torch.nn.init.constant_(m.bias,31.0)
def main():
	# Load the configuration from the YAML file
	# 2024-10-20, A.M.
	# config = load_config('scripts/scripts/config.yaml')
	config = load_config('config.yaml')

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
	#PDL_features = [0]*14
	#ETL_features = [0]*14
	#PDL_features = [0]*1
	#ETL_features = [0]*1
	for i in range(14):
	# for i in [0]:
	# for i in [0,1]:
		for j in range(14):
		# for j in [1]:
		# for j in [1,0]:
			firstIndex= i
			secondIndex= j
			print('(i=',i,', j=',j,'): ',feature_names[i],'*',feature_names[j])
			Xf1_PDL_1, Xf1_ETL_1= get_feature(firstIndex,data_path= data_path)
			AllValues= np.concatenate((np.transpose(Xf1_PDL_1),np.transpose(Xf1_ETL_1)),axis=1)
			Std= np.std(AllValues)
			#print("Mean=",np.mean(AllValues))
			#print("Std=",Std)
			Xf1_PDL_1= Xf1_PDL_1 / Std
			Xf1_ETL_1= Xf1_ETL_1 / Std
			Xf1_PDL_2, Xf1_ETL_2= get_feature(secondIndex,data_path= data_path)
			AllValues= np.concatenate((np.transpose(Xf1_PDL_2),np.transpose(Xf1_ETL_2)),axis=1)
			Std= np.std(AllValues)
			#print("Mean=",np.mean(AllValues))
			#print("Std=",Std)
			Xf1_PDL_2= Xf1_PDL_2 / Std
			Xf1_ETL_2= Xf1_ETL_2 / Std
			N= len(Xf1_PDL_1)
			#print(N)
			#print(Xf1_PDL_1)
			#print(Xf1_PDL_2)
			Xf1_PDL= Xf1_PDL_1
			Xf1_ETL= Xf1_ETL_1
			#print("Xf1_PDL_2.size=",Xf1_PDL_2.size)
			#print("Xf1_PDL_2.shape=",Xf1_PDL_2.shape)
			Xf1_PDL_2_size= Xf1_PDL_2.shape[0]
			for k in range(Xf1_PDL_2.shape[0]):
				for m in range(Xf1_PDL_2.shape[1]):
					# print(k)
					Xf1_PDL[k][m]= -1 * Xf1_PDL[k][m] * Xf1_PDL_2[k][m]
			for k in range(Xf1_ETL_2.shape[0]):
				for m in range(Xf1_ETL_2.shape[1]):
					Xf1_ETL[k][m]= -1 * Xf1_ETL[k][m] * Xf1_ETL_2[k][m]
			#Xf1_PDL, Xf1_ETL= CreateData()
			# Convert to numpy arrays if they aren't already
			Xf1_PDL = np.array(Xf1_PDL)  # Shape: (15, 600)
			Xf1_ETL = np.array(Xf1_ETL)  # Shape: (21, 600)
			PDL_features= Xf1_PDL
			ETL_features= Xf1_ETL
			#print("ETL_features.shape:",ETL_features.shape)
			#print("ETL_features:",ETL_features)
			# counter= counter + 1
			PDL_features = np.array(PDL_features)  # Shape (14, 15, 600)
			ETL_features = np.array(ETL_features)  # Shape (14, 21, 600)
			# Reshape them into (14, 15*600) and (14, 21*600)
			#PDL_reshaped = PDL_features.reshape(14, -1)  # Shape becomes (14, 15*600)
			#ETL_reshaped = ETL_features.reshape(14, -1)  # Shape becomes (14, 21*600)
			PDL_reshaped= PDL_features.reshape(1,-1)
			ETL_reshaped= ETL_features.reshape(1,-1)
			#PDL_reshaped = PDL_features.reshape(1,-1)
			#ETL_reshaped = ETL_features.reshape(1,-1)
			#print("***PDL_reshaped:",PDL_reshaped.shape)
			#  Concatenate along axis 1 (the last axis)
			combined_features = np.concatenate((PDL_reshaped, ETL_reshaped), axis=1)  # Shape becomes (14, 15*600 + 21*600)
			#
			#print("Combined features:", combined_features)
			#
			## Step 3: Create labels (y) with 0 for PDL and 1 for ETL
			y_PDL = np.ones(PDL_reshaped.shape[1], dtype=int)  # Labels 0 for PDL
			y_ETL = np.zeros(ETL_reshaped.shape[1], dtype=int)   # Labels 1 for ETL
			#
			## Concatenate the labels
			y = np.concatenate((y_PDL, y_ETL))  # Labels of length N (2*14 = 28 in this case)
			#combined_features= PDL_reshaped
			# y= ETL_reshaped.reshape(100,1)
			#y= ETL_reshaped.reshape(100,1)
			# Output shapes
			#print("Combined features shape:", combined_features.shape)  # Shape (14, 15*600 + 21*600)
			#print("y shape:", y.shape)
			#print("y:", y)
			#  Convert numpy arrays to torch tensors
			features_tensor= torch.tensor(combined_features, dtype=torch.float32).T   # Convert features to tensor
			labels_tensor= torch.tensor(y, dtype=torch.float)  # Convert labels to tensor
			#print("labels_tensor shape:",labels_tensor.shape)
			#print("features_tensor:", features_tensor)
			#print("labels_tensor:", labels_tensor)
			#  Create a TensorDataset
			dataset = TensorDataset(features_tensor, labels_tensor)
			# dataset= list(zip(Xf1_PDL,Xf1_ETL))
			# Create a DataLoader with an option for batching
			batch_size = 1 # batch_size  # Define your batch size
			dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
			#for x, y in dataloader:
			#	print('= x y =')
			#	print(x.shape)
			#	print(y.shape)
			#	break
			#model=SimpleBinaryClassifier(input_dim=14)
			#model= Alexey_net_Step(input_size=1,output_size=1)
			#model= Alexey_net_Rect_Vec(input_size=1,output_size=1)
			#model= LeftRightLinear(input_features=1,output_features=1)
			#model.apply(init_weights)
			model= SimpleBinaryClassifier(input_dim=1)
			train_model(model, dataloader, num_epochs=num_epochs, path=save_path+'/alexei_1.pth', learning_rate=learning_rate)
			model= torch.load(save_path+'/alexei_1.pth', weights_only=False)
			# now let's try to run the tests
			#PDL_features =  np.swapaxes(np.swapaxes(PDL_features, 0, 1), 1,2)
			#ETL_features =  np.swapaxes(np.swapaxes(ETL_features, 0, 1),1,2)
			#PDL_features= np.swapaxes(PDL_features, 0, 1)
			#ETL_features= np.swapaxes(ETL_features, 0, 1)
			#print("PDL_features:",PDL_features)
			#pdl_classif= viz.viz_quantity_features(PDL_features, 'PDL', model,save_path+'/PDL_quant_1_')
			#print("ETL_features:",ETL_features)
			#etl_classif= viz.viz_quantity_features(ETL_features, 'ETL', model,save_path+'/ETL_quant_1_')

if __name__ == "__main__":
	main()
