import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def save_feature_scatter_plot(train_loader, save_path: str, feat1_name='f1', feat2_name='f2'):
    """
    Generates a 2D scatter plot of features from the dataset and saves it to the specified path.

    :param train_loader: DataLoader object containing the features and labels.
    :param save_path: The file path where the plot will be saved.
    """
    # Initialize lists to collect features and labels
    all_features = []
    all_labels = []

    # Iterate through the DataLoader to collect all features and labels
    for features, labels in train_loader:
        all_features.append(features)
        all_labels.append(labels)

    # Concatenate all features and labels into single tensors
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Convert to numpy for plotting
    features_np = all_features.numpy()
    labels_np = all_labels.numpy()

    # Plot the scatter plot for the whole dataset
    plt.scatter(features_np[:, 0], features_np[:, 1], c=labels_np, cmap='bwr', alpha=0.6)

    # Add colorbar to show the mapping of colors to labels
    plt.colorbar(ticks=[0, 1])

    # Set the labels and title
    plt.xlabel(feat1_name)
    plt.ylabel(feat2_name)
    plt.title('2D Feature Scatter Plot with Binary Labels: 1 is PD and 0 is ET')

    # Save the plot to the specified path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    # Clear the current figure to free up memory
    plt.clf()


def plot_probabilistic_output(outputs,save_path,feat1_name='f1', feat2_name='f2'):
    # Example data
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

    # Plot the outputs
    outputs = outputs.reshape(xx.shape)
    plt.contourf(xx, yy, outputs, levels=50, cmap='coolwarm', alpha=0.6)  # Use 'coolwarm' cmap for red to blue gradient
    plt.colorbar()
    plt.xlabel(feat1_name)
    plt.ylabel(feat2_name)
    plt.title('Network Output')
    plt.savefig(save_path)
    # Clear the current figure to free up memory
    plt.clf()

def viz_quantity_features(X_features, label, model, save_path):
    import matplotlib.pyplot as plt

    # Store results for each patient
    patient_like_counts = []
    # Store results for visualization
    pd_counts = []
    et_counts = []
    print(f'input shape {X_features.shape}')

    for patient_idx, patient_data in enumerate(X_features):
        pd_like_count = 0
        et_like_count = 0
        print(patient_data.shape)

        for sample_idx, sample in enumerate(patient_data):
            sample_tensor = torch.from_numpy(sample).float().unsqueeze(0)
            # print(sample_tensor.shape)
            with torch.no_grad():
                prediction = model(sample_tensor)

            if prediction.item() > 0.5:
                pd_like_count += 1
            else:
                et_like_count += 1

        pd_counts.append(pd_like_count)
        et_counts.append(et_like_count)

        patient_like_counts.append((pd_like_count, et_like_count))

        print(f"Patient {patient_idx + 1}: {label}")
        print(f"  PD-like samples: {pd_like_count}")
        print(f"  ET-like samples: {et_like_count}")
        print("\n")

    # Plotting
    patients = range(1, len(X_features) + 1)

    plt.figure(figsize=(10, 6))
    plt.bar(patients, pd_counts, color='red', label='PD-like')
    plt.bar(patients, et_counts, color='blue', bottom=pd_counts, label='ET-like')
    plt.xlabel('Patient Number')
    plt.ylabel('Number of Samples')
    plt.title(f'PD-like vs ET-like Sample Counts per Patient for {label} patients')
    plt.legend()
    plt.xticks(patients)
    # Save the plot to the specified path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    # Clear the current figure to free up memory
    plt.clf()
    patient_like_counts = np.array(patient_like_counts)# Convert the patient_like_counts to a numpy array
    return patient_like_counts
