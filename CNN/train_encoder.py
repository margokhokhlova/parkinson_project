import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from customdata_loader import Dataset
from autoencoder import  Autoencoder
from autoencoder_check import plot_latent


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

data_raw = pd.read_csv("CNN/data_cnn.csv",header = None)
data_np = data_raw.to_numpy()
labels = data_np[:,0]
values = data_np[:,1:]
max_epochs = 1000
# print(labels)
total_step = len(values)

# Parameters
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 0}


# Generators
training_set = Dataset(values, labels)
training_generator = torch.utils.data.DataLoader(training_set, **params)


# model

latent_dims = 2
n_features=24
autoencoder = Autoencoder(n_features, latent_dims).to(device) # GPU
model = autoencoder.float()
# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# training
loss_avg_list = []
# Loop over epochs
for epoch in range(max_epochs):
    correct_sum = 0
    # Training
    loss_list_epoch = []
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = torch.unsqueeze(local_batch,1).to(device).float(), local_labels.to(device)
        # print(local_batch.shape, local_labels.shape)
        # Run the forward pass
        outputs = model(local_batch.float())
        loss = criterion(outputs, local_batch)
        loss_list_epoch.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss {np.mean(loss_list_epoch)}")
    loss_avg_list.append(np.mean(loss_list_epoch))

plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(loss_avg_list)
plt.show()
plt.savefig('train_loss_avg.png')

plot_latent(model, data = training_generator, num_batches=1)