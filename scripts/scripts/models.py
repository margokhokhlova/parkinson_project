import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchviz import make_dot

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



#  Multi-head model
# Define N Separate Heads: Each head will have its own set of fully connected (FC) layers for feature extraction.
# Merge Features: After extracting features from each head, concatenate them into a single vector.
# Final Binary Prediction: The merged features will pass through another layer, and a final binary prediction will be made using the sigmoid activation function.



class MultiHeadNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_heads=3):
        super(MultiHeadNN, self).__init__()

        self.num_heads = num_heads
        
        # Define three heads, each with its own fully connected layers
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_heads)
        ])
        
        # Layer to combine features from all heads
        self.fc_merge = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
        # Final output layer for binary classification
        self.fc_final = nn.Linear(hidden_dim, 1)

    def forward(self, features_list):
            """
            Args:
                features_list: A list or tuple of tensors, where each tensor is the input
                            for a specific head. Each tensor should have the shape (batch_size, input_dim).
                            Example: [x1, x2, x3] for three heads.
            """
            assert len(features_list) == self.num_heads, f"Expected {self.num_heads} feature inputs, but got {len(features_list)}"
            
            # Pass each feature input through its corresponding head
            head_features = [self.heads[i](features_list[i]) for i in range(self.num_heads)]
            
            # Concatenate features from all heads
            combined_features = torch.cat(head_features, dim=1)
            
            # Pass the combined features through the merging layer
            merged = F.relu(self.fc_merge(combined_features))
            
            # Final binary prediction
            output = torch.sigmoid(self.fc_final(merged))
            
            return output
    

# # Initialize model
model = MultiHeadNN(input_dim=1, hidden_dim=16, num_heads=3)

# Example inputs for each head (3 different feature sets, each with shape (batch_size, input_dim))
x1 = torch.randn(16,1)  # Feature set for head 1
x2 = torch.randn(16,1)  # Feature set for head 2
x3 = torch.randn(16,1)  # Feature set for head 3

# Forward pass (list of features for each head)
output = model([x1, x2, x3])
print(output)
graphtree = make_dot(output.mean(), params=dict(model.named_parameters()),show_attrs=True, show_saved=True)

graphtree.render("graphtree_output", format="png")