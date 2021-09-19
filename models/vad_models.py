import torch

import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, inpud_dim=40, hidden_dim=64, n_layers=2, dropout=0.5):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(inpud_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, 1)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x
    
class FeedForwardModel(nn.Module):
    def __init__(self, inpud_dim=40, hidden_dim=64, dropout=0.5):
        super(FeedForwardModel, self).__init__()

        self.fc1 = nn.Linear(inpud_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x