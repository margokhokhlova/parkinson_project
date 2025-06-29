import numpy as np
from data_outils import dataloader
from data_outils.data_features import *
from data_outils.data_perm import *
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import pandas as pd
import sys


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
        test_accuracy = []
        for X_trainval, y_trainval, X_test, y_test, (i, j) in leave_one_pair_out(PDL_features, ETL_features):
            print(f"Left out PDL[{i}] and ETL[{j}]")
            validation_accuracy = []           
            # Do the validation parameters only once, with one pair out, and select the best parameter 
            if i==0 and j==0 : 
                for valp, num_kernels in enumerate([1_00, 1_000, 10_000]):
                    best_val = 0.0
                    for i in range(num_val_cycles):
                        X_train, y_train, X_val, y_val = stratified_random_split(X_trainval, y_trainval, val_size=2)
                        # Normalize
                        normalizer = FeatNormalisation(num_feat=len(model_features))
                        normalizer.calculate_norm_values(X_train)
                        X_train = normalizer.normalize(X_train)
                        X_val = normalizer.normalize(X_val)

                        # Transpose to (samples, timepoints, channels)
                        X_train = X_train.transpose(0, 2, 1)
                        X_val = X_val.transpose(0, 2, 1)

                        # Convert to sktime format
                        X_train_df = to_sktime_format(X_train)
                        X_val_df = to_sktime_format(X_val)

                        # Fit Rocket and transform
                        rocket = Rocket(num_kernels=num_kernels)
                        rocket.fit(X_train_df)
                        X_train_transformed = rocket.transform(X_train_df)
                        X_val_transformed = rocket.transform(X_val_df)

                        # Train classifier
                        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
                        classifier.fit(X_train_transformed, y_train)
                        y_pred = classifier.predict(X_val_transformed)
                        acc = accuracy_score(y_val, y_pred)
                        print(f"Validation Accuracy: {acc:.4f}")
                        validation_accuracy.append(acc)
                    print(f"Rocket, nkernels is {num_kernels} Validation Accuracy mean:{np.mean(validation_accuracy)}, {np.std(validation_accuracy)}")
                    if np.mean(validation_accuracy)>best_val:
                        best_val = np.mean(validation_accuracy)
                        optimal_kernel = num_kernels
                print(f"Final best test parameter is : {optimal_kernel} with val binary accuracy {best_val}")
            # Now do the test check, on all the data pairs
            normalizer = FeatNormalisation(num_feat=len(model_features))
            normalizer.calculate_norm_values(X_trainval)
            X_train = normalizer.normalize(X_trainval)
            X_val = normalizer.normalize(X_test)

            # Transpose to (samples, timepoints, channels)
            X_train = X_train.transpose(0, 2, 1)
            X_val= X_val.transpose(0, 2, 1)

            # Convert to sktime format
            X_train_df = to_sktime_format(X_train)
            X_val_df = to_sktime_format(X_val)

   
            # Fit Rocket and transform
            rocket = Rocket(num_kernels=optimal_kernel)
            rocket.fit(X_train_df)
            X_train_transformed = rocket.transform(X_train_df)
            X_val_transformed = rocket.transform(X_val_df)

             # Train classifier
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            classifier.fit(X_train_transformed, y_trainval)
            y_pred = classifier.predict(X_val_transformed)
            acc = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy: {acc:.4f}")
            if acc is not None and not isinstance(acc, str):
                if isinstance(acc, np.ndarray):
                    acc = acc.item()  # Converts array([0.91]) to 0.91
                test_accuracy.append(acc)
            else:
                print(f"Test accuracy is None = {acc} {y_pred, y_test}")
  
        print(f"Rocket, nkernels is {num_kernels}, combination {model_features} Test Accuracy mean:{np.mean(test_accuracy)}, {np.std(test_accuracy)}")
        print(test_accuracy, "\n")

if __name__ == "__main__":
	main()