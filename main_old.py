import numpy as np
from data_outils import dataloader
from data_outils.data_features import *
from data_outils.data_perm import *
from code.rocket_functions import generate_kernels, apply_kernels
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score

def main():
    # parameters
    min_len_established = 600
    num_val_cycles = 100
    model_features= range(14)
    data_path = '/home/margokat/Documents/projects/margo_files/data/data_lstm_august24_PDET_left_right.csv'
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
    # make iterations over all set with one pair leave out  (14 × 14 = 196 total combinations).
    # Loop over leave-one-out combinations
    for X_trainval, y_trainval, X_test, y_test, (i, j) in leave_one_pair_out(PDL_features, ETL_features):
        print(f"Left out PDL[{i}] and ETL[{j}]")
        # this is an internal loop to select the best validation parameters for methods which have parameters to adjust
        for i in range(num_val_cycles):
            X_train, y_train, X_val, y_val = stratified_random_split(X_trainval, y_trainval, val_size=2)
            # Normalize
            normalizer = FeatNormalisation()
            normalizer.calculate_norm_values(X_train)
            X_train = normalizer.normalize(X_train)
            X_val = normalizer.normalize(X_val)

            # Transpose to (samples, timepoints, channels)
            X_train = X_train.transpose(0, 2, 1)
            X_val = X_val.transpose(0, 2, 1)

            n_samples, n_timepoints, n_channels = X_train.shape

            # Apply kernels independently per channel, then concatenate features
            all_train_features = []
            all_val_features = []

            for c in range(n_channels):
                kernels = generate_kernels(input_length=n_timepoints, num_kernels=10)
                X_train_c = X_train[:, :, c]
                X_val_c = X_val[:, :, c]

                train_features = apply_kernels(X_train_c, kernels)
                val_features = apply_kernels(X_val_c, kernels)

                all_train_features.append(train_features)
                all_val_features.append(val_features)

            # Concatenate all channel features
            X_train_transformed = np.concatenate(all_train_features, axis=1)
            X_val_transformed = np.concatenate(all_val_features, axis=1)

            # Train classifier
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            classifier.fit(X_train_transformed, y_train)
            y_pred = classifier.predict(X_val_transformed)
            acc = accuracy_score(y_val, y_pred)
            print(f"Validation Accuracy: {acc:.4f}")
            exit()


if __name__ == "__main__":
	main()