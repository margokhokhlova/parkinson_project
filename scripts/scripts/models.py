import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
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

def train_model(model, train_loader, num_epochs=10, path=None, learning_rate=0.001):
	criterion= nn.BCELoss()  # Binary Cross Entropy Loss
	# optimizer= optim.Adam(model.parameters(), lr=0.001)
	optimizer= optim.Adam(model.parameters(), lr=learning_rate) # 2024-10-30, A.M.

	for epoch in range(num_epochs):
		model.train()
		running_loss = 0.0
		correct_predictions = 0
		total_samples = 0
		for data, labels in train_loader:
			optimizer.zero_grad()
			outputs = model(data)
			#outputs = model(data) #.squeeze()  # Flatten output to match target shape
			#print(outputs, labels)
			loss = criterion(outputs, labels.unsqueeze(1))
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			# Calculate accuracy
			predicted_labels = (outputs >= 0.5).float()  # Convert probabilities to binary predictions (0 or 1)
			correct_predictions += (predicted_labels == labels.unsqueeze(1)).sum().item()  # Compare with reshaped labels
			total_samples += labels.size(0)
		epoch_loss = running_loss / len(train_loader)
		epoch_accuracy = correct_predictions / total_samples
		if epoch%2 == 1:
			print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
	if path is not None:
		torch.save(model, path)
	return epoch_accuracy
# Multi-head model
# Define N Separate Heads: Each head will have its own set of fully connected (FC) layers for feature extraction.
# Merge Features: After extracting features from each head, concatenate them into a single vector.
# Final Binary Prediction: The merged features will pass through another layer, and a final binary prediction will be made using the sigmoid activation function.

class MultiHeadNN(nn.Module):
	def __init__(self, input_dim, hidden_dim=16, num_heads=3):
		super(MultiHeadNN, self).__init__()

		self.num_heads = num_heads
		
		# Define a head for each feature dimension, each with its own fully connected layers
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

	def forward(self, features):
		"""
		Args:
			features: Tensor of shape (batch_size, num_heads), where each column corresponds
				to the input feature for a specific head.
		"""
		batch_size = features.size(0)
		
		# Split features into list of tensors for each head
		split_features = torch.split(features, 1, dim=1)  # Splits into [batch_size, 1] tensors
		
		# Pass each split feature through its corresponding head
		head_features = [self.heads[i](split_features[i]) for i in range(self.num_heads)]
		
		# Concatenate features from all heads along the feature dimension
		combined_features = torch.cat(head_features, dim=1)
		
		# Pass the combined features through the merging layer
		merged = F.relu(self.fc_merge(combined_features))
		
		# Final binary prediction
		output = torch.sigmoid(self.fc_final(merged))
		
		return output

# # Initialize model
model = MultiHeadNN(input_dim=1, hidden_dim=16, num_heads=14)

# Example input with batch size of 1 and 14 heads
x = torch.randn(1, 14)  # One batch with 14 features (1x14)

# Forward pass through the model
output = model(x)
print("Model: ",output)

class StepActivationFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx,x):
		result= torch.zeros(x.shape)
		result[(x <= 0.0)]= 0.0
		result[(x >  0.0)]= 1.0
		return result
	@staticmethod
	def backward(ctx,grad_output):
		grad_input= grad_output.clone()
		if (grad_input > -1.0) & (grad_input < 1.0):
			return None
		return grad_input

class LinearFunction(torch.autograd.Function):
	@staticmethod
	# ctx is the first argument to forward
	def forward(ctx, input, weight, bias=None):
		# The forward pass can use ctx.
		ctx.save_for_backward(input, weight, bias)
		# A.M.: output = input.mm(weight.t())
		output = input * weight
		if bias is not None:
			output += bias.unsqueeze(0).expand_as(output)
		return output
	@staticmethod
	def backward(ctx, grad_output):
		input, weight, bias = ctx.saved_tensors
		grad_input = grad_weight = grad_bias = None
		if ctx.needs_input_grad[0]:
			grad_input = grad_output.mm(weight)
		if ctx.needs_input_grad[1]:
			grad_weight = grad_output.t().mm(input)
		if bias is not None and ctx.needs_input_grad[2]:
			grad_bias = grad_output.sum(0)
		return grad_input, grad_weight, grad_bias


class Alexey_net_Step(torch.nn.Module):
	# Constructor
	def __init__(self,input_size,output_size):
		super(Alexey_net_Step,self).__init__()
		# hidden layer
		self.hidden= output_size
		self.shared_layer1= LeftRightLinear(input_size,output_size)
		self.activation_output_1= StepActivationFunction()
	# prediction function
	def forward(self,x):
		batch_size, n_wavetrains= x.shape
		# output= torch.zeros(batch_size,self.hidden,device=x.device)
		# for i in range(n_wavetrains):
		# 	# Select the feature across all examples in the batch
		#feature= x[:,i].unsqueeze(1)  # Shape: (batch_size,1)
		# Apply the shared linear layer
		fo12= self.shared_layer1(x)
		output= self.activation_output_1.apply(fo12)
		return output
		
class LeftRightLinear(torch.nn.Module):
	constant_left_weight= torch.as_tensor(-1.0)
	constant_right_weight= torch.as_tensor(-1.0)
	activation_middle_1= StepActivationFunction()
	def __init__(self, input_features, output_features, bias=True):
		super().__init__()
		self.input_features = input_features
		self.output_features = output_features
		# nn.Parameter is a special kind of Tensor, that will get
		# automatically registered as Module's parameter once it's assigned
		# as an attribute. Parameters and buffers need to be registered, or
		# they won't appear in .parameters() (doesn't apply to buffers), and
		# won't be converted when e.g. .cuda() is called. You can use
		# .register_buffer() to register buffers.
		# nn.Parameters require gradients by default.
		# A.M.:
		# self.weight = torch.nn.Parameter(torch.empty(output_features, input_features))
		self.register_parameter('weight', None)
		if bias:
			self.leftBias = torch.nn.Parameter(torch.empty(output_features))
			self.rightBias = torch.nn.Parameter(torch.empty(output_features))
		else:
			# You should always register all possible parameters, but the
			# optional ones can be None if you want.
			self.register_parameter('leftBias', None)
			self.register_parameter('rihtBias', None)
		# Not a very smart way to initialize weights
		# A.M.: torch.nn.init.uniform_(self.weight, -0.1, 0.1)
		# if self.bias is not None:
		#	torch.nn.init.uniform_(self.bias, -0.1, 0.1)
	def forward(self, input):
		if self.leftBias.data < self.rightBias.data:
			fo1 = LinearFunction.apply(input, self.constant_left_weight, self.leftBias)
			fo1 = self.activation_middle_1.apply(-fo1)
			fo2 = LinearFunction.apply(input, self.constant_right_weight, self.rightBias)
			fo2= self.activation_middle_1.apply(fo2)
			result= (fo1 * fo2)
		else:
			fo1= LinearFunction.apply(input, self.constant_left_weight, self.leftBias)
			fo1= self.activation_middle_1.apply(fo1)
			fo2= LinearFunction.apply(input, self.constant_right_weight, self.rightBias)
			fo2= self.activation_middle_1.apply(-fo2)
			result= (fo1 * fo2)
		if self.leftBias.data == self.rightBias.data:
			if input == self.leftBias.data:
				result+= 1.0
		return result
	def extra_repr(self):
		# it by printing an object of this class.
		return 'input_features={}, output_features={}, leftBias={}, rightBias={}'.format(
			self.input_features, self.output_features, self.leftBias, self.rightBias is not None)

class Alexey_net_Rect_Vec(torch.nn.Module):
	# Constructor
	def __init__(self,input_size,output_size):
		super(Alexey_net_Rect_Vec,self).__init__()
		# hidden layer
		self.hidden= output_size
		self.shared_layer1= LeftRightLinear(input_size,output_size)
		self.activation_output_1= StepActivationFunction()
	# prediction function
	def forward(self,x):
		# print('Alexey_net_Rect_Vec.forward.x=',x)
		batch_size, n_wavetrains= x.shape
		#print('batch_size=',batch_size)
		#print('n_wavetrains=',n_wavetrains)
		# output= torch.zeros(batch_size,self.hidden,device=x.device)
		output= torch.zeros(n_wavetrains,1,device=x.device)
		# output= torch.zeros(1,1,device=x.device)
		for i in range(n_wavetrains):
			# Select the feature across all examples in the batch
			feature= x[:,i].unsqueeze(1)  # Shape: (batch_size,1)
			# Apply the shared linear layer
			fo12= self.shared_layer1(feature)
			# print('fo12=',fo12)
			# Sum the output for each feature
			# output += fo12
			output[i]= fo12
		# output= output / n_wavetrains
		# output= self.activation_output_1.apply(output)
		# print('output=',output)
		return output
		# return output[:,0]
		# return torch.transpose(output.unsqueeze(0),1,0)

class MultiHeadNN_Alex(nn.Module):
	def __init__(self, input_dim, hidden_dim=1, num_heads=3):
		super(MultiHeadNN_Alex, self).__init__()

		self.num_heads = num_heads

		# Define a head for each feature dimension, each will use Alexey_net_Step as a custom model
		self.heads = nn.ModuleList([
			Alexey_net_Step(input_dim, hidden_dim) for _ in range(num_heads)
		])

		# Layer to combine features from all heads
		self.fc_merge = nn.Linear(hidden_dim * num_heads, hidden_dim)

		# Final output layer for binary classification
		self.fc_final = nn.Linear(hidden_dim, 1)

	def forward(self, features):
		"""
		Args:
			features: Tensor of shape (batch_size, num_heads), where each column corresponds
				to the input feature for a specific head.
		"""
		batch_size = features.size(0)
		
		# Split features into list of tensors for each head
		split_features = torch.split(features, 1, dim=1)  # Splits into [batch_size, 1] tensors
		
		# Pass each split feature through its corresponding head
		head_features = [self.heads[i](split_features[i]) for i in range(self.num_heads)]
		
		# Concatenate features from all heads along the feature dimension
		combined_features = torch.cat(head_features, dim=1)
		
		# Pass the combined features through the merging layer
		merged = F.relu(self.fc_merge(combined_features))
		
		# Final binary prediction
		output = torch.sigmoid(self.fc_final(merged))
		
		return output


# Here's a straightforward model with a fully connected layer 
# that maps the 14 input values to a single output for binary 
# classification. The model consists of an input layer with 14 neurons
# (one for each input feature) and an output layer with a single neuron, 
# followed by a sigmoid activation for binary classification.

class SimpleBinaryClassifier(nn.Module):
	def __init__(self, input_dim=14):
		super(SimpleBinaryClassifier, self).__init__()
		
		# A simple linear layer that takes 14 input features and outputs a single value
		self.fc = nn.Linear(input_dim, 1)
		
	def forward(self, features):
		"""
		Args:
			features: Tensor of shape (batch_size, 14), where each column corresponds
				to an input feature.
		"""
		# Apply the fully connected layer
		output = self.fc(features)
		
		# Apply sigmoid activation for binary classification
		output = torch.sigmoid(output)
		
		return output

class EnhancedBinaryClassifier(nn.Module):
    def __init__(self, input_dim=14, hidden_dims=[32, 16]):
        super(EnhancedBinaryClassifier, self).__init__()

        # Sequentially define the layers
        layers = []
        current_dim = input_dim

        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))  # Linear layer
            layers.append(nn.ReLU())  # Non-linear activation
            layers.append(nn.Dropout(0.3))  # Dropout for regularization
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())  # Sigmoid activation for binary classification

        self.model = nn.Sequential(*layers)

    def forward(self, features):
        """
        Args:
            features: Tensor of shape (batch_size, input_dim).
        """
        # Pass through the model
        output = self.model(features)
        return output
	


class MultiHeadPairsNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_heads=91, num_layers=1, merge_opt = 'merge'):
        super(MultiHeadPairsNN, self).__init__()
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Define a head for each feature dimension with a variable number of layers
        self.heads = nn.ModuleList([
            self._build_head(input_dim, hidden_dim, num_layers) for _ in range(num_heads)
        ])
        
        # Layer to combine features from all heads
        self.fc_merge = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
        # Final output layer for binary classification
        self.fc_final = nn.Linear(hidden_dim, 1)
        self.merge_opt = merge_opt
        self.head_weights = nn.Parameter(torch.ones(self.num_heads))  # Learnable weights


    
    def _build_head(self, input_dim, hidden_dim, num_layers):
        """
        Helper function to build a head with the specified number of layers.
        """
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def forward(self, x):
		# Split the input into 91 pairs along the last dimension
		# Each pair has shape [N, 2]
        pair_inputs = torch.unbind(x, dim=-1)  # This creates a list of 91 tensors of shape [N, 2]

        # Pass input through each head
        head_outputs = [head(pair_input) for head, pair_input in zip(self.heads, pair_inputs)]

        # Concatenate outputs from all heads - only for concat, skipped if log sum
        merged_output = torch.cat(head_outputs, dim=1)
		# Combine the list of tensors into a single tensor along the head dimension
        head_outputs = torch.stack(head_outputs, dim=1)  # Shape: [batch_size, num_heads, hidden_dim]
        if self.merge_opt == 'logsum':
			# Compute the sum of logarithms across the `num_heads` dimension
            log_outputs = torch.log(1 + head_outputs)  # Add 1 to avoid log(0)
            summed_logs = torch.sum(log_outputs, dim=1)  # Sum over the `num_heads` dimension
			# Final classification
            return torch.sigmoid(self.fc_final(summed_logs))
        elif self.merge_opt == 'merge':
        # # # Merge features
            merged_output = self.fc_merge(merged_output)
            return torch.sigmoid(self.fc_final(merged_output))
        elif self.merge_opt == 'weighted':
			# Apply learnable weights to the head outputs
            weighted_outputs = torch.sum(head_outputs * self.head_weights.view(1, -1, 1), dim=1)  # Shape: [batch_size, hidden_dim]
            # Final classification
            return torch.sigmoid(self.fc_final(weighted_outputs))

	# def forward(self, features):
	# 	"""
	# 	Args:
	# 		features: Tensor of shape (batch_size, num_heads), where each column corresponds
	# 			to the input feature for a specific head.
	# 	"""
	# 	batch_size = features.size(0)
		
	# 	# Split features into list of tensors for each head
	# 	split_features = torch.split(features, 1, dim=1)  # Splits into [batch_size, 1] tensors
		
	# 	# Pass each split feature through its corresponding head
	# 	head_features = [self.heads[i](split_features[i]) for i in range(self.num_heads)]
		
	# 	# Concatenate features from all heads along the feature dimension
	# 	combined_features = torch.cat(head_features, dim=1)
		
	# 	# Pass the combined features through the merging layer
	# 	merged = F.relu(self.fc_merge(combined_features))
		
	# 	# Final binary prediction
	# 	output = torch.sigmoid(self.fc_final(merged))
		
	# 	return output