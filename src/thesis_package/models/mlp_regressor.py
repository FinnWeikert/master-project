import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from collections import deque
from sklearn.base import BaseEstimator, RegressorMixin

class PyTorchMLPEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, hidden_dim=16, n_hidden=1, dropout=0.1, lr=5e-4, 
                 max_epochs=1000, batch_size=10, weight_decay=1e-4, 
                 n_models=4, patience=50, avg_window=50, device='cpu', seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.n_models = n_models        # Bagging: Number of independent models
        self.patience = patience        # Epochs to wait for Train Loss improvement
        self.avg_window = avg_window    # Snapshots for Weight Averaging
        self.device = device
        self.models_ = []
        self.seed = seed
        
    def _build_single_model(self, seed):
        torch.manual_seed(seed) # Ensure different init for each ensemble member
        layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.LeakyReLU(0.01), nn.Dropout(self.dropout)]
        for _ in range(self.n_hidden - 1):
            layers.extend([nn.Linear(self.hidden_dim, (self.hidden_dim)//2), nn.LeakyReLU(0.01), nn.Dropout(self.dropout)])
        layers.append(nn.Linear((self.hidden_dim)//(2**(self.n_hidden-1)), 1))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y):
        self.models_ = []
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        for i in range(self.n_models):
            # 1. Initialize unique model for Bagging
            model = self._build_single_model(seed=i*self.seed)
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            criterion = nn.MSELoss()
            
            # 2. Setup Data
            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            best_train_loss = float('inf')
            epochs_no_improve = 0
            state_buffer = deque(maxlen=self.avg_window)

            # 3. Training Loop with Training-Loss-based Early Stopping
            for epoch in range(self.max_epochs):
                model.train()
                epoch_loss = 0.0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(batch_X), batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_train_loss = epoch_loss / len(loader)
                state_buffer.append(deepcopy(model.state_dict()))

                # Check for convergence/stagnation
                if avg_train_loss < best_train_loss - 1e-4:
                    best_train_loss = avg_train_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    break

            # 4. Weight Averaging (Smooths the final local minimum)
            if len(state_buffer) > 1:
                avg_state = self._average_weights(list(state_buffer))
                model.load_state_dict(avg_state)
            
            self.models_.append(model)
            
        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        all_preds = []
        
        for model in self.models_:
            model.eval()
            with torch.no_grad():
                all_preds.append(model(X_tensor).cpu().numpy().flatten())
        
        # 5. Aggregate predictions (Bagging)
        return np.mean(all_preds, axis=0)

    def _average_weights(self, state_dicts):
        avg_state = {}
        for key in state_dicts[0].keys():
            avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
        return avg_state