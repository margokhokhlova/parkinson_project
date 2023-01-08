import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from customdata_loader import Dataset
from models import ConvNet1D



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

data_raw = pd.read_csv("CNN/data_cnn.csv",header = None)
data_np = data_raw.to_numpy()
labels = data_np[:,0]
values = data_np[:,1:]
max_epochs = 500
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
model = ConvNet1D()
model = model.float()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# training
loss_list = []
acc_list = []
acc_list_epoch = []
# Loop over epochs
for epoch in range(max_epochs):
    correct_sum = 0
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = torch.unsqueeze(local_batch,1).to(device), local_labels.to(device)
        # print(local_batch.shape, local_labels.shape)
        # Run the forward pass
        outputs = model(local_batch.float())
        loss = criterion(outputs, torch.max(local_labels, 1)[1])
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         # Track the accuracy
        total = local_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        _, actual = torch.max(local_labels, 1)
        correct = (predicted == actual).sum().item()
        correct_sum = correct_sum + (correct/total)
        acc_list.append(correct / total)
    print("Epoch")
    print(epoch)
    print("accuracy")
    print(correct_sum/int(np.floor(total_step/params['batch_size'])))
    acc_list_epoch.append(correct_sum/int(np.floor(total_step/params['batch_size'])))

plt.title('Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(acc_list_epoch)
plt.show()
plt.savefig('train_accuracy.png')

