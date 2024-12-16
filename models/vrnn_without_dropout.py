import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

# Define the VRNN model
class VRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, n_layers, n_classes):
        super(VRNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        # Feature extraction (phi_x)
        self.phi_x = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Encoder Network (for variational part)
        self.enc = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),  # Concatenate phi_x_t and h_t-1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(hidden_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.Softplus()  # Ensure positive standard deviation
        )

        # Prior (latent random variable distribution)
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(hidden_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.Softplus()  # Ensure positive standard deviation
        )
        
        # Decoder Network (to generate output based on z_t and h_t-1)
        self.dec = nn.Sequential(
            nn.Linear(z_dim + hidden_dim, hidden_dim),  # Concatenate z_t and h_t-1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.dec_mean = nn.Linear(hidden_dim, input_dim)
        
        # Final classification layer (if needed)
        self.fc = nn.Linear(hidden_dim, n_classes)  # For classification

        # RNN (GRU in this case)
        self.rnn = nn.GRU(z_dim + hidden_dim, hidden_dim, n_layers)
        
    def forward(self, x, sample_posterior=True):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden state
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Losses
        kld_loss = 0
        nll_loss = 0
    
        # Loop over each timestep
        for t in range(seq_len):
            phi_x_t = self.phi_x(x[:, t, :])  # Shape: (batch_size, hidden_dim)
            
            # Encoder step (for VAE at each timestep)
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], dim=1))  # Concatenate phi_x_t and h[-1]
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)
            
            # Sampling from the posterior using reparameterization trick if sample_posterior is True
            if sample_posterior:
                z_t = self.reparameterized_sample(enc_mean_t, enc_std_t)
            else:
                # Use the mean of the posterior as a deterministic z_t during inference
                z_t = enc_mean_t
            
            # Prior distribution at timestep t
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            
            # Compute KL divergence (between posterior and prior)
            kld_loss += self.kl_divergence(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            
            # Decoder step (generating x_t)
            dec_input = torch.cat([z_t, h[-1]], dim=1)  # Concatenate z_t and h[-1]
            dec_t = self.dec(dec_input)  # Output shape: (batch_size, hidden_dim)
            dec_mean_t = self.dec_mean(dec_t)  # Output shape: (batch_size, input_dim)
            
            # Compute negative log-likelihood (NLL) loss using CrossEntropyLoss
            nll_loss += F.cross_entropy(dec_mean_t, x[:, t, :].argmax(dim=1))  # Change this line to CrossEntropyLoss.
            
            # Update hidden state using RNN
            rnn_input = torch.cat([phi_x_t, z_t], dim=1).unsqueeze(0)  # Add batch dimension
            _, h = self.rnn(rnn_input, h)  # Output shape: (n_layers, batch_size, hidden_dim)
    
        # Final classification output (if needed)
        class_output = self.fc(h[-1])  # Shape: (batch_size, n_classes)
        
        return class_output, kld_loss, nll_loss

    def reparameterized_sample(self, mean, std):
        """Reparameterization trick to sample from the Gaussian distribution"""
        eps = torch.randn_like(std)
        return eps * std + mean
    
    def kl_divergence(self, mean_1, std_1, mean_2, std_2):
        """KL Divergence for Gaussian Distributions"""
        kl = torch.log(std_2 + 1e-8) - torch.log(std_1 + 1e-8) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / (2 * std_2.pow(2)) - 0.5
        return torch.sum(kl, dim=1).mean()  # Average over batch and sum over dimensions
    
    def nll_loss(self, mean, x):
        """Negative Log-Likelihood for Continuous Outputs (using MSE)"""
        return nn.MSELoss()(mean, x)

# Load and preprocess the dataset
def load_and_preprocess_data(file_path, test_size=0.2, max_features=1000):
    df = pd.read_csv(file_path)
    # df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert 'ham'/'spam' to 0/1
    
    # Split the dataset into training and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Tokenizing and vectorizing the text
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_df['tweet']).toarray()
    y_train = train_df['label'].values
    X_test = vectorizer.transform(test_df['tweet']).toarray()
    y_test = test_df['label'].values
    
    print(f"X_train shape after TF-IDF vectorization: {X_train.shape}")
    print(f"X_test shape after TF-IDF vectorization: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, y_train, X_test, y_test, vectorizer

# Dataset class
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Add seq_len dimension with length 1
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Bayesian Inference
import torch
import numpy as np
import torch.nn.functional as F

def bayesian_inference_with_posterior(model, dataloader, device, num_samples=10):
    model.eval()
    all_preds = []
    all_uncertainties = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            
            # Store predictions and uncertainties
            sampled_preds = []
            for _ in range(num_samples):
                outputs, _, _ = model(inputs, sample_posterior=True)  # Sample from posterior during inference
                sampled_preds.append(outputs.cpu().numpy())
            
            # Aggregate the predictions (majority vote for classification)
            sampled_preds = np.stack(sampled_preds, axis=0)  # Shape: (num_samples, batch_size, num_classes)
            mean_preds = sampled_preds.mean(axis=0)  # Shape: (batch_size, num_classes)
            uncertainty = sampled_preds.std(axis=0)  # Shape: (batch_size, num_classes)

            # Get final predictions by argmax
            final_preds = np.argmax(mean_preds, axis=1)
            all_preds.append(final_preds)
            all_uncertainties.append(uncertainty)

    all_preds = np.concatenate(all_preds, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)

    return all_preds, all_uncertainties



# Main script
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VRNN(input_dim=input_dim, hidden_dim=128, z_dim=64, n_layers=2, n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_kld_loss = 0
        total_nll_loss = 0
        total_class_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_output, kld_loss, nll_loss = model(inputs)  # No need to add seq_len dimension
        
            # Calculate the classification loss
            class_loss = nn.CrossEntropyLoss()(class_output, labels)
            
            # Total loss = classification loss + kld loss + nll loss
            loss = class_loss + 0.1 * kld_loss + 0.1 * nll_loss

            loss.backward()
            optimizer.step()
            
            total_class_loss += class_loss.item()
            total_kld_loss += kld_loss.item()
            total_nll_loss += nll_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(class_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Class Loss: {total_class_loss/len(train_dataloader):.4f}, "
              f"KLD Loss: {total_kld_loss/len(train_dataloader):.4f}, NLL Loss: {total_nll_loss/len(train_dataloader):.4f})
    
      # Bayesian Inference on Test Data
    test_preds, test_uncertainties = bayesian_inference(model, test_dataloader, device, num_samples=10)
    
    # Calculate test accuracy
    test_labels = y_test
    test_acc = (test_preds == test_labels).mean() * 100
    print(f"Test Accuracy: {test_acc:.2f}%")
