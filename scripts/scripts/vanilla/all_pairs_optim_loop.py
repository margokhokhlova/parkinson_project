import sys
sys.path.append('.') # 2024-10-20, A.M.

import matplotlib.pyplot as plt
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
import optim

def load_config(config_file):
	with open(config_file, 'r') as file:
		config= yaml.safe_load(file)
	return config

def main():
	#
	integration_mode= 'concatenate'
	# integration_mode= 'add_logarithm'
	#
	# print("Start the experiment...")
	# Load the configuration from the YAML file
	# 2024-10-20, A.M.
	# config= load_config('scripts/scripts/config.yaml')
	config= load_config('config.yaml')
	#
	# Access the settings from the config
	learning_rate= config['settings']['learning_rate']
	batch_size= config['settings']['batch_size']
	num_epochs= config['settings']['num_epochs']
	model_name= config['settings']['model_name']
	data_path= config['settings']['data_path']
	combinations= config['settings']['possible_pairs']
	save_path= config['settings']['save_path']
	feature_names= config['settings']['feat_order']
	#
	save_path= save_path + "_pairs_2"
	#
	# Print out the configuration
	print(f"Integration Mode: {integration_mode}")
	print(f"Learning Rate: {learning_rate}")
	print(f"Batch Size: {batch_size}")
	print(f"Number of Epochs: {num_epochs}")
	print(f"Model Name: {model_name}")
	print(f"Data Path: {data_path}")
	print(f"Feature_names: {feature_names}")
	#
	# Initialize lists to store the features and labels
	initial_PDL_features= [0]*14
	initial_ETL_features= [0]*14
	#
	for j in range(14):
		j_Xf1_PDL,j_Xf1_ETL= get_feature(j,data_path= data_path)
		# Convert to numpy arrays if they aren't already
		j_Xf1_PDL= np.array(j_Xf1_PDL)  # Shape: (15, 600)
		j_Xf1_ETL= np.array(j_Xf1_ETL)  # Shape: (21, 600)
		initial_PDL_features[j]= j_Xf1_PDL
		initial_ETL_features[j]= j_Xf1_ETL
	initial_PDL_features= np.array(initial_PDL_features)  # Shape (14, 15, 600)
	initial_ETL_features= np.array(initial_ETL_features)  # Shape (14, 21, 600)
	initial_PDL_features= initial_PDL_features.transpose(1,0,2)
	initial_ETL_features= initial_ETL_features.transpose(1,0,2)
	#
	accuracies= []
	#
	all_mean_accuracy= []
	all_median_accuracy= []
	all_std_accuracy= []
	all_min_accuracy= []
	all_max_accuracy= []
	#
	step= 0;
	#
	# Run the experiment 1000 times on different train and test sets
	for _ in range(1000):
		#
		step= step + 1
		#
		print("***** ",step," *****")
		#
		# Split the dataset into training and testing sets
		step_X_train,step_X_test= train_test_split(initial_PDL_features,test_size=0.2)
		step_y_train,step_y_test= train_test_split(initial_ETL_features,test_size=0.2)
		#
		step_X_train= step_X_train.transpose(1,0,2)
		step_y_train= step_y_train.transpose(1,0,2)
		step_X_test= step_X_test.transpose(1,0,2)
		step_y_test= step_y_test.transpose(1,0,2)
		#
		step_PDL_features= step_X_train
		step_ETL_features= step_y_train
		#
		all_pdl_features= []
		all_etl_features= []
		#
		number_of_pair= 0
		#
		for pair in combinations:
			#
			number_of_pair= number_of_pair + 1
			number_of_combinations= len(combinations)
			print("     Train ",step," [",number_of_pair,"/",number_of_combinations,"]: ",f'Feature pair: {feature_names[pair[0]]} - {feature_names[pair[1]]}')
			#
			new_path= save_path+f'/{pair[0]}_{pair[1]}'
			# print(f'Starting to train for the first pair {pair[0]}_{pair[1]}, results are saved {new_path}')
			if not os.path.exists(new_path):
				os.makedirs(new_path)
			#
			Xf1_PDL= step_PDL_features[pair[0]][:][:];
			Xf1_ETL= step_ETL_features[pair[0]][:][:];
			Xf2_PDL= step_PDL_features[pair[1]][:][:];
			Xf2_ETL= step_ETL_features[pair[1]][:][:];
			#
			#print("Xf1_PDL:",np.array(Xf1_PDL).shape)
			#print("Xf1_ETL:",np.array(Xf1_ETL).shape)
			#print("Xf2_PDL:",np.array(Xf2_PDL).shape)
			#print("Xf2_ETL:",np.array(Xf2_ETL).shape)
			#
			first_tensor,second_tensor= stack_features(Xf1_PDL,Xf2_PDL,Xf1_ETL,Xf2_ETL)
			#
			#print("first_tensor:",np.array(first_tensor).shape)
			#print("second_tensor:",np.array(second_tensor).shape)
			#
			dataset= TensorDataset(first_tensor,second_tensor)
			train_loader= DataLoader(dataset,batch_size=batch_size,shuffle=True)
			# viz.save_feature_scatter_plot(train_loader, new_path+'/scatter_plot.png',feature_names[pair[0]], feature_names[pair[1]])
			# model training and all that
			model= SimpleNN(2,16) #just 2 features
			train_model(model,train_loader,num_epochs=num_epochs,learning_rate=learning_rate,path=new_path+f'/{feature_names[pair[0]]}_{feature_names[pair[1]]}.pth')
			# train_model(model,train_loader,num_epochs=num_epochs,learning_rate=learning_rate)
			# plotting and training results
			# min_f1, max_f1, min_f2, max_f2= get_min_max_values(Xf1_PDL,Xf2_PDL, Xf1_ETL,Xf2_ETL)
			# xx,yy= np.meshgrid(np.linspace(min_f1,max_f1, 100),np.linspace(min_f2,max_f2,100))
			# grid= np.c_[xx.ravel(), yy.ravel()]
			# grid_tensor= torch.tensor(grid, dtype=torch.float32) 
			# Get the network outputs for the grid and plot the output
			# print("***** Get the network outputs for the grid and plot the output *****")
			# with torch.no_grad():
			#	outputs= model(grid_tensor).numpy()   
			# viz.plot_probabilistic_output(outputs,new_path+f'/proba_{feature_names[pair[0]]}_{feature_names[pair[1]]}')
			# finally, plot the resulting histograms per number of features
			# print("***** finally, plot the resulting histograms per number of features *****")
			X_PDL,X_ETL= get_per_patient_features(Xf1_PDL,Xf2_PDL,Xf1_ETL,Xf2_ETL)
			#print("first_tensor:",np.array(X_PDL).shape)
			#print("second_tensor:",np.array(X_ETL).shape)
			pdl_features= viz.viz_quantity_features(X_PDL,'PDL',model,new_path+f'/PDL_quant_{feature_names[pair[0]]}_{feature_names[pair[1]]}')
			etl_features= viz.viz_quantity_features(X_ETL,'ETL',model,new_path+f'/ETL_quant_{feature_names[pair[0]]}_{feature_names[pair[1]]}')
			#print("AFTER viz_quantity_features: pdl_features:",np.array(pdl_features).shape)
			#print("AFTER viz_quantity_features: etl_features:",np.array(etl_features).shape)
			#pdl_features= viz.viz_quantity_features(X_PDL,'PDL',model)
			#etl_features= viz.viz_quantity_features(X_ETL,'ETL',model)
			# print("pdl_features:",pdl_features)
			# print("etl_features:",etl_features)
			# try to classify them
			# print("***** try to classify them *****")
			pdl_features= np.array([[pdl_features[i][0]] for i in range(len(pdl_features))])
			etl_features= np.array([[etl_features[i][0]] for i in range(len(etl_features))])
			# print("pdl_features:",pdl_features)
			# print("etl_features:",etl_features)
			#print("GET FIRST COLUMN: pdl_features.shape(): ",np.array(pdl_features).shape);
			#print("GET FIRST COLUMN: etl_features.shape(): ",np.array(etl_features).shape);
			if (len(all_pdl_features)==0):
				# all_pdl_features= pdl_features**2
				# all_pdl_features= np.log(pdl_features+1)
				# all_pdl_features= pdl_features
				# all_pdl_features= pdl_features**2
				if integration_mode=='concatenate':
					all_pdl_features= pdl_features
				elif integration_mode=='add_logarithm':
					all_pdl_features= np.log(pdl_features+1)
			else:
				#for k in range(len(all_pdl_features)):
				#	# all_pdl_features[k]= all_pdl_features[k] + pdl_features[k]**2
				#	# all_pdl_features[k]= all_pdl_features[k] + np.log(pdl_features[k]+1)
				#	# all_pdl_features[k]= all_pdl_features[k] + pdl_features[k]
				#	# all_pdl_features[k]= all_pdl_features[k] + pdl_features[k]**2
				#	# all_pdl_features[k]= all_pdl_features[k] + np.log(pdl_features[k]+1)
				if integration_mode=='concatenate':
					all_pdl_features= np.hstack((all_pdl_features,pdl_features))
				elif integration_mode=='add_logarithm':
					for k in range(len(all_pdl_features)):
						all_pdl_features[k]= all_pdl_features[k] + np.log(pdl_features[k]+1)
			if (len(all_etl_features)==0):
				# all_etl_features= etl_features**2
				# all_etl_features= np.log(etl_features+1)
				# all_etl_features= etl_features
				# all_etl_features= etl_features**2
				if integration_mode=='concatenate':
					all_etl_features= etl_features
				elif integration_mode=='add_logarithm':
					all_etl_features= np.log(etl_features+1)
			else:
				#for k in range(len(all_etl_features)):
				#	# all_etl_features[k]= all_etl_features[k] + etl_features[k]**2
				#	# all_etl_features[k]= all_etl_features[k] + np.log(etl_features[k]+1)
				#	# all_etl_features[k]= all_etl_features[k] + etl_features[k]
				#	# all_etl_features[k]= all_etl_features[k] + etl_features[k]**2
				#	# all_etl_features[k]= all_etl_features[k] + np.log(etl_features[k]+1)
				if integration_mode=='concatenate':
					all_etl_features= np.hstack((all_etl_features,etl_features))
				elif integration_mode=='add_logarithm':
					for k in range(len(all_etl_features)):
						all_etl_features[k]= all_etl_features[k] + np.log(etl_features[k]+1)
			# ENF OF "for pair in combinations"
			#print("all_pdl_features: ",all_pdl_features);
			#print("all_etl_features: ",all_etl_features);
			#print("all_pdl_features.shape(): ",np.array(all_pdl_features).shape);
			#print("all_etl_features.shape(): ",np.array(all_etl_features).shape);
		#
		#print("* all_pdl_features.shape(): ",np.array(all_pdl_features).shape);
		#print("* all_etl_features.shape(): ",np.array(all_etl_features).shape);
		#
		# X_quant= np.vstack((pdl_features,  etl_features))
		X_quant= np.vstack((np.array(all_pdl_features),np.array(all_etl_features)))
		#
		print("X_quant.shape(): ",np.array(X_quant).shape);
		# print("X_quant: ",X_quant);
		# print("X_quant[1][1]: ",X_quant[1][1]);
		#
		# Create labels: 1 for PD-like, 0 for ET-like
		y_pdl= np.ones(pdl_features.shape[0]) # Labels for PD-like patients
		y_etl= np.zeros(etl_features.shape[0]) # Labels for ET-like patients
		#
		# Concatenate the labels
		y_quant= np.concatenate((y_pdl, y_etl))
		#
		print("y_quant.shape(): ",np.array(y_quant).shape);
		#
		# Initialize and train the classifier (Logistic Regression in this example)
		# classifier= LogisticRegression(max_iter=10_000)
		# classifier.fit(X_quant,y_quant)
		#
		Weights= optim.choose_weights(X_quant,y_quant);
		# Weights= [1.0,0.0,0.0]
		#
		print("Weights: ",Weights)
		#
		X_quant_shape_0= X_quant.shape[0]
		X_quant_shape_1= X_quant.shape[1]
		#matrix= X_quant
		#for k in range(X_quant_shape_1):
		#	for m in range(X_quant_shape_0):
		#		matrix[m][k]= matrix[m][k] * Weights[k]
		vector= np.zeros((X_quant_shape_0,1))
		for k in range(X_quant_shape_0):
			# print("k=",k)
			sum= 0
			for m in range(X_quant_shape_1):
				# print("k=",k," m=",m," X_quant[k][m]=",X_quant[k][m])
				A= X_quant[k][m] * Weights[m]
				sum= sum + np.log(A+1)
				# sum= sum + A
			vector[k][0]= sum
		# print("vector:",vector)
		classifier= LogisticRegression(max_iter=10_000)
		# classifier.fit(matrix,y_quant)
		classifier.fit(vector,y_quant)
		#
		################################################################
		# TEST                                                         #
		################################################################
		#
		step_PDL_features= step_X_test
		step_ETL_features= step_y_test
		#step_PDL_features= step_X_train
		#step_ETL_features= step_y_train
		#
		all_pdl_features= []
		all_etl_features= []
		#
		number_of_pair= 0
		#
		for pair in combinations:
			#
			number_of_pair= number_of_pair + 1
			number_of_combinations= len(combinations)
			print("     Test ",step," [",number_of_pair,"/",number_of_combinations,"]: ",f'Feature pair: {feature_names[pair[0]]} - {feature_names[pair[1]]}')
			#
			new_path= save_path+f'/{pair[0]}_{pair[1]}'
			# print(f'Starting to train for the first pair {pair[0]}_{pair[1]}, results are saved {new_path}')
			# if not os.path.exists(new_path):
			#	os.makedirs(new_path)
			#
			Xf1_PDL= step_PDL_features[pair[0]][:][:];
			Xf1_ETL= step_ETL_features[pair[0]][:][:];
			Xf2_PDL= step_PDL_features[pair[1]][:][:];
			Xf2_ETL= step_ETL_features[pair[1]][:][:];
			#
			#first_tensor,second_tensor= stack_features(Xf1_PDL,Xf2_PDL,Xf1_ETL,Xf2_ETL)
			#
			#dataset= TensorDataset(first_tensor,second_tensor)
			#train_loader= DataLoader(dataset,batch_size=batch_size,shuffle=True)
			#model= SimpleNN(2,16) # just 2 features
			model= torch.load(new_path+f'/{feature_names[pair[0]]}_{feature_names[pair[1]]}.pth')
			#train_model(model,train_loader,num_epochs=num_epochs,learning_rate=learning_rate)
			X_PDL,X_ETL= get_per_patient_features(Xf1_PDL,Xf2_PDL,Xf1_ETL,Xf2_ETL)
			pdl_features= viz.viz_quantity_features(X_PDL,'PDL',model,new_path+f'/PDL_quant_{feature_names[pair[0]]}_{feature_names[pair[1]]}')
			etl_features= viz.viz_quantity_features(X_ETL,'ETL',model,new_path+f'/ETL_quant_{feature_names[pair[0]]}_{feature_names[pair[1]]}')
			#pdl_features= viz.viz_quantity_features(X_PDL,'PDL',model)
			#etl_features= viz.viz_quantity_features(X_ETL,'ETL',model)
			# print("pdl_features:",pdl_features)
			# print("etl_features:",etl_features)
			# try to classify them
			# print("***** try to classify them *****")
			pdl_features= np.array([[pdl_features[i][0]] for i in range(len(pdl_features))])
			etl_features= np.array([[etl_features[i][0]] for i in range(len(etl_features))])
			# print("pdl_features:",pdl_features)
			# print("etl_features:",etl_features)
			if (len(all_pdl_features)==0):
				# all_pdl_features= pdl_features**2
				# all_pdl_features= np.log(pdl_features+1)
				# all_pdl_features= pdl_features
				# all_pdl_features= pdl_features**2
				if integration_mode=='concatenate':
					all_pdl_features= pdl_features
				elif integration_mode=='add_logarithm':
					all_pdl_features= np.log(pdl_features+1)
			else:
				#for k in range(len(all_pdl_features)):
				#	# all_pdl_features[k]= all_pdl_features[k] + pdl_features[k]**2
				#	# all_pdl_features[k]= all_pdl_features[k] + np.log(pdl_features[k]+1)
				#	# all_pdl_features[k]= all_pdl_features[k] + pdl_features[k]
				#	# all_pdl_features[k]= all_pdl_features[k] + pdl_features[k]**2
				#	# all_pdl_features[k]= all_pdl_features[k] + np.log(pdl_features[k]+1)
				if integration_mode=='concatenate':
					all_pdl_features= np.hstack((all_pdl_features,pdl_features))
				elif integration_mode=='add_logarithm':
					for k in range(len(all_pdl_features)):
						all_pdl_features[k]= all_pdl_features[k] + np.log(pdl_features[k]+1)
			if (len(all_etl_features)==0):
				# all_etl_features= etl_features**2
				# all_etl_features= np.log(etl_features+1)
				# all_etl_features= etl_features
				# all_etl_features= etl_features**2
				if integration_mode=='concatenate':
					all_etl_features= etl_features
				elif integration_mode=='add_logarithm':
					all_etl_features= np.log(etl_features+1)
			else:
				#for k in range(len(all_etl_features)):
				#	# all_etl_features[k]= all_etl_features[k] + etl_features[k]**2
				#	# all_etl_features[k]= all_etl_features[k] + np.log(etl_features[k]+1)
				#	# all_etl_features[k]= all_etl_features[k] + etl_features[k]
				#	# all_etl_features[k]= all_etl_features[k] + etl_features[k]**2
				#	# all_etl_features[k]= all_etl_features[k] + np.log(etl_features[k]+1)
				if integration_mode=='concatenate':
					all_etl_features= np.hstack((all_etl_features,etl_features))
				elif integration_mode=='add_logarithm':
					for k in range(len(all_etl_features)):
						all_etl_features[k]= all_etl_features[k] + np.log(etl_features[k]+1)
			# ENF OF "for pair in combinations"
		#
		X_quant= np.vstack((np.array(all_pdl_features),np.array(all_etl_features)))
		#
		# print("X_quant.shape(): ",np.array(X_quant).shape);
		#
		# Create labels: 1 for PD-like, 0 for ET-like
		y_pdl= np.ones(pdl_features.shape[0]) # Labels for PD-like patients
		y_etl= np.zeros(etl_features.shape[0]) # Labels for ET-like patients
		#
		# Concatenate the labels
		y_quant= np.concatenate((y_pdl,y_etl))
		#
		################################################################
		#
		X_quant_shape_0= X_quant.shape[0]
		X_quant_shape_1= X_quant.shape[1]
		#matrix= X_quant
		#for k in range(X_quant_shape_1):
		#	for m in range(X_quant_shape_0):
		#		matrix[m][k]= matrix[m][k] * Weights[k]
		vector= np.zeros((X_quant_shape_0,1))
		for k in range(X_quant_shape_0):
			sum= 0
			for m in range(X_quant_shape_1):
				A= X_quant[k][m] * Weights[m]
				sum= sum + np.log(A+1)
				# sum= sum + A
			vector[k][0]= sum
		#
		# Predict on the test set
		# y_pred= classifier.predict(matrix)
		y_pred= classifier.predict(vector)
		#
		# Evaluate the classifier
		accuracy= accuracy_score(y_quant,y_pred)
		#
		# print(f"     Classification accuracy: {accuracy:.3f}")
		#
		accuracies.append(accuracy)
		#
		# Calculate the mean and standard deviation of the accuracies
		mean_accuracy= np.mean(accuracies)
		median_accuracy= np.median(accuracies)
		std_accuracy= np.std(accuracies)
		min_accuracy= np.min(accuracies)
		max_accuracy= np.max(accuracies)
		#
		all_mean_accuracy= np.hstack((all_mean_accuracy,mean_accuracy))
		all_median_accuracy= np.hstack((all_median_accuracy,median_accuracy))
		all_std_accuracy= np.hstack((all_std_accuracy,std_accuracy))
		all_min_accuracy= np.hstack((all_min_accuracy,min_accuracy))
		all_max_accuracy= np.hstack((all_max_accuracy,max_accuracy))
		#
		plt.close("all")
		#
		viz.plot_two_curves(all_median_accuracy,all_mean_accuracy,"r-","b-","Joint Median vs. Joint Mean","joint_median","joint_mean",save_path)
		viz.plot_curve(all_median_accuracy,"r-","Joint Median","joint_median",save_path)
		viz.plot_curve(all_mean_accuracy,"b-","Joint Mean","joint_mean",save_path)
		viz.plot_curve(all_min_accuracy,"m-","Joint Min","joint_min",save_path)
		viz.plot_curve(all_max_accuracy,"g-","Joint Max","joint_max",save_path)
		viz.plot_std_curve(all_std_accuracy,"c-","Joint Std","joint_std",save_path)
		viz.plot_curve(accuracies,"k-","Accuracies","accuracies",save_path)
	# Print out the configuration
	print(f"Integration Mode: {integration_mode}")
	print(f"Learning Rate: {learning_rate}")
	print(f"Batch Size: {batch_size}")
	print(f"Number of Epochs: {num_epochs}")
	print(f"Model Name: {model_name}")
	print(f"Data Path: {data_path}")
	print(f"Feature_names: {feature_names}")

if __name__ == "__main__":
	main()
