import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# Define the handwritten model (as done before)
class TwoLayerNNHandwritten(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, is_classification=False):
        super(TwoLayerNNHandwritten, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) if is_classification else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.softmax:
            x = self.softmax(x)
        return x

# Define the built-in PyTorch model for comparison
class TwoLayerNNBuiltIn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, is_classification=False):
        super(TwoLayerNNBuiltIn, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) if is_classification else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.softmax:
            x = self.softmax(x)
        return x

# Loss function and optimizer
def get_loss_and_optimizer(model, is_classification=False):
    if is_classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return criterion, optimizer

# Training loop for handwritten and built-in models
def train_model(model, X_train, y_train, epochs=100, is_classification=False):
    criterion, optimizer = get_loss_and_optimizer(model, is_classification=is_classification)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    # Log for training
    epoch_losses = []
    epoch_times = []

    for epoch in range(epochs):
        start_epoch_time = time.time()
        
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Log the loss and epoch time
        epoch_losses.append(loss.item())
        epoch_times.append(time.time() - start_epoch_time)
        
        # Print the loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Epoch Time: {epoch_times[-1]:.4f}s")

    return epoch_losses, epoch_times

# Function to test the model (calculate predictions)
def test_model(model, X_test):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_test_tensor)
    return predictions

# Function to compare performance
def compare_models(X_train, y_train, X_test, epochs=100, is_classification=False):
    print("\nTraining Handwritten Model:")
    # Handwritten model
    handwritten_model = TwoLayerNNHandwritten(input_dim=X_train.shape[1], hidden_dim=5, output_dim=1 if not is_classification else 2, is_classification=is_classification)
    start_time = time.time()
    handwritten_loss, handwritten_epoch_times = train_model(handwritten_model, X_train, y_train, epochs, is_classification)
    handwritten_time = time.time() - start_time
    handwritten_predictions = test_model(handwritten_model, X_test)

    print(f"\nHandwritten Model - Total Training Time: {handwritten_time:.4f}s")

    print("\nTraining Built-In Model:")
    # Built-in model
    built_in_model = TwoLayerNNBuiltIn(input_dim=X_train.shape[1], hidden_dim=5, output_dim=1 if not is_classification else 2, is_classification=is_classification)
    start_time = time.time()
    built_in_loss, built_in_epoch_times = train_model(built_in_model, X_train, y_train, epochs, is_classification)
    built_in_time = time.time() - start_time
    built_in_predictions = test_model(built_in_model, X_test)

    print(f"\nBuilt-in Model - Total Training Time: {built_in_time:.4f}s")

    # Print final loss and comparison
    print("\nTraining Summary:")
    print(f"Handwritten Model - Final Loss: {handwritten_loss[-1]:.4f}")
    print(f"Built-in Model - Final Loss: {built_in_loss[-1]:.4f}")
    
    # Compare epoch times (average time per epoch)
    handwritten_avg_epoch_time = np.mean(handwritten_epoch_times)
    built_in_avg_epoch_time = np.mean(built_in_epoch_times)
    print(f"Handwritten Model - Average Epoch Time: {handwritten_avg_epoch_time:.4f}s")
    print(f"Built-in Model - Average Epoch Time: {built_in_avg_epoch_time:.4f}s")

    return handwritten_predictions, built_in_predictions, handwritten_loss, built_in_loss, handwritten_time, built_in_time

# Example Data for Comparison
X_train = np.random.rand(100, 3)  # 100 samples, 3 features
y_train = np.random.rand(100, 1)  # 100 regression targets
X_test = np.random.rand(10, 3)    # 10 test samples

# Perform the comparison
handwritten_preds, built_in_preds, handwritten_loss, built_in_loss, handwritten_time, built_in_time = compare_models(X_train, y_train, X_test, epochs=100, is_classification=False)
