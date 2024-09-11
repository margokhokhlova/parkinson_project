import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the simpler model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        
        self.fc3 = nn.Linear(hidden_dim, 1)  # Single output for binary classification

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Use sigmoid activation for binary classification
        return x

# # Example usage
# batch_size, input_dim = 32, 128  # Adjust these as needed
# model = SimpleNN(input_dim)

def train_model(model, train_loader, num_epochs=10, path='results/model.pth'):
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            outputs = model(data).squeeze()  # Flatten output to match target shape
            #print(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
             # Calculate accuracy
            predicted_labels = (outputs >= 0.5).float()  # Convert probabilities to binary predictions
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        if epoch%100 == 1:
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    torch.save(model, path)