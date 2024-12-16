import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Simple RNN for Text Classification
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_classes):
        super(SimpleRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        # RNN layer
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)  # Initialize hidden state
        
        # RNN forward pass
        out, _ = self.rnn(x, h0)  # out shape: (batch_size, seq_len, hidden_dim)
        
        # Use the last hidden state for classification
        out = out[:, -1, :]  # Take the output of the last time step (batch_size, hidden_dim)
        
        # Pass through the fully connected layer
        out = self.fc(out)  # Output shape: (batch_size, n_classes)
        
        return out

# Clean labels function to ensure no invalid values
def clean_labels(labels):
    labels = labels.astype(int)  # Convert to integer type if needed
    labels[labels == -1] = 0     # Replace -1 with 0 (or another value if appropriate)
    return labels

# Load and preprocess the dataset
def load_and_preprocess_data(file_path, test_size=0.2, max_features=1000):
    df = pd.read_csv(file_path)

    # Clean the labels
    df['label'] = clean_labels(df['label'])  # Apply the cleaning function to the labels

    # Split the dataset into training and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Tokenizing and vectorizing the text
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_df['tweet']).toarray()
    y_train = train_df['label'].values
    X_test = vectorizer.transform(test_df['tweet']).toarray()
    y_test = test_df['label'].values
    
    # Check for label distribution (should only have 0 or 1 for binary classification)
    print("Label distribution in training set:")
    print(pd.Series(y_train).value_counts())

    print("Label distribution in test set:")
    print(pd.Series(y_test).value_counts())
    
    return X_train, y_train, X_test, y_test, vectorizer

# Dataset class
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Add seq_len dimension
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Main script for comparison
if __name__ == "__main__":
    file_path = '/kaggle/input/datsettextclassicfication/sentiment_analysis.csv'
    X_train, y_train, X_test, y_test, vectorizer = load_and_preprocess_data(file_path)
    
    # Adjust the input dimension based on the actual number of features
    input_dim = X_train.shape[1]
    
    # Create datasets and dataloaders with fixed batch size
    batch_size = 16
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define model, optimizer, and criterion
    device = torch.device("cpu")  # Force the model to use CPU
    model = SimpleRNN(input_dim=input_dim, hidden_dim=128, n_layers=2, n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_output = model(inputs)  # (batch_size, n_classes)
            
            # Classification loss
            loss = nn.CrossEntropyLoss()(class_output, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(class_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader):.4f}")
    
    # Test the model
    model.eval()
    test_preds = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            class_output = model(inputs)
            _, predicted = torch.max(class_output, 1)
            test_preds.append(predicted.cpu().numpy())
    
    test_preds = np.concatenate(test_preds, axis=0)
    test_acc = (test_preds == y_test).mean() * 100
    print(f"Test Accuracy: {test_acc:.2f}%")
