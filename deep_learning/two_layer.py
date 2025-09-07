import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Two-Layer Neural Network model
class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, is_classification=False):
        super(TwoLayerNN, self).__init__()
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

# Training loop
def train_model(model, X_train, y_train, epochs=100):
    criterion, optimizer = get_loss_and_optimizer(model, is_classification=False) 
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Test the model
def test_model(model, X_test):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_test_tensor)
    return predictions

# Example for Regression
X_train = np.random.rand(100, 3)
y_train = np.random.rand(100, 1)
X_test = np.random.rand(10, 3)

model_regression = TwoLayerNN(input_dim=3, hidden_dim=5, output_dim=1, is_classification=False)
train_model(model_regression, X_train, y_train, epochs=100)
predictions_regression = test_model(model_regression, X_test)
print("Predictions for Regression Task:", predictions_regression)

# Example for Classification
X_train_class = np.random.rand(100, 3)
y_train_class = np.random.randint(0, 2, size=(100, 1))
X_test_class = np.random.rand(10, 3)

model_classification = TwoLayerNN(input_dim=3, hidden_dim=5, output_dim=2, is_classification=True)
train_model(model_classification, X_train_class, y_train_class, epochs=100)
predictions_classification = test_model(model_classification, X_test_class)
print("Predictions for Classification Task:", predictions_classification)
