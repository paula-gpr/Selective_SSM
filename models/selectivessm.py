import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence

import pdb

# Custom PyTorch Dataset
class EHRDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ts_values = torch.tensor(item['ts_values'], dtype=torch.float32)
        ts_indicators = torch.tensor(item['ts_indicators'], dtype=torch.float32)
        ts_times = torch.tensor(item['ts_times'], dtype=torch.float32)
        static = torch.tensor(item['static'], dtype=torch.float32)
        label = torch.tensor(item['labels'], dtype=torch.long)
        return ts_values, ts_indicators, ts_times, static, label
    
# Custom collate_fn for handling variable-length sequences
def custom_collate_fn(batch):
    ts_values = [item[0].clone().detach() for item in batch]
    ts_indicators = [item[1].clone().detach() for item in batch]
    ts_times = [item[2].clone().detach() for item in batch]
    static_features = torch.stack([item[3].clone().detach() for item in batch])
    labels = torch.stack([item[4] for item in batch]).to(torch.long)
    
    # Pad sequences to the same length
    ts_values_padded = pad_sequence(ts_values, batch_first=True)
    ts_indicators_padded = pad_sequence(ts_indicators, batch_first=True)
    ts_times_padded = pad_sequence(ts_times, batch_first=True)
    
    # Create lengths for masking
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)
    
    return ts_values_padded, ts_indicators_padded, ts_times_padded, static_features, labels, lengths


# SelectiveSSM Model
class SelectiveSSM(nn.Module):
    def __init__(self, ts_dim, static_dim, state_dim, hidden_dim, num_classes):
        super(SelectiveSSM, self).__init__()
        self.fc_F = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * state_dim)
        )
        self.fc_B = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * ts_dim)
        )
        self.attention = nn.Sequential(
            nn.Linear(ts_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )


    def forward(self, ts_values, ts_indicators, ts_times, static_features, lengths):
        batch_size, seq_len, ts_dim = ts_values.shape

        batch_size = static_features.size(0)  # Número de ejemplos en el lote
        state_dim = int(self.fc_F[-1].out_features ** 0.5)  # Calcular state_dim dinámicamente

        # Compute matrices F, B from static features
        F = self.fc_F(static_features).view(batch_size, state_dim, state_dim)
        B = self.fc_B(static_features).view(batch_size, state_dim, -1)

        # Initialize state
        state = torch.zeros(batch_size, F.shape[1], device=ts_values.device)
        
        # Create mask to handle variable lengths
        mask = (torch.arange(seq_len, device=ts_values.device)
                .unsqueeze(0).expand(batch_size, -1) < lengths.unsqueeze(1)).float()
        
        ts_values = ts_values * mask.unsqueeze(-1)
        
        # Iterate over time steps
        for t in range(seq_len):
            ts_step = ts_values[:, t, :]  # [batch_size, ts_dim]
            mask_step = mask[:, t].unsqueeze(-1)  # [batch_size, 1]
            ts_step_masked = ts_step * mask_step  # [batch_size, ts_dim]

            attention_weights = self.attention(ts_step_masked) + 1e-8 # [batch_size, 1]
            ts_selected = ts_step_masked * attention_weights  # Element-wise multiplication
            
            # Update state
            state = torch.bmm(F, state.unsqueeze(-1)).squeeze(-1)
            control_effect = torch.bmm(B, ts_selected.unsqueeze(-1)).squeeze(-1)
            state = state + control_effect
        
        # Classify using final state
        logits = self.classifier(state)
        return logits


# Training Function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for ts_values, ts_indicators, ts_times, static_features, labels, lengths in train_loader:
        ts_values, ts_indicators, ts_times, static_features, labels, lengths = (
            ts_values.to(device),
            ts_indicators.to(device),
            ts_times.to(device),
            static_features.to(device),
            labels.to(device),
            lengths.to(device),
        )
        optimizer.zero_grad()
        outputs = model(ts_values, ts_indicators, ts_times, static_features, lengths)
        loss = criterion(outputs, labels)
        # pdb.set_trace()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        #pdb.set_trace()
    return total_loss / len(train_loader)


# Evaluation Function
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_with_metrics(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for ts_values, ts_indicators, ts_times, static_features, labels, lengths in loader:
            ts_values, ts_indicators, ts_times, static_features, labels, lengths = (
                ts_values.to(device),
                ts_indicators.to(device),
                ts_times.to(device),
                static_features.to(device),
                labels.to(device),
                lengths.to(device),
            )
            outputs = model(ts_values, ts_indicators, ts_times, static_features, lengths)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for class 1
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    return acc, auroc, auprc


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight) 
        if m.bias is not None:
            nn.init.zeros_(m.bias)  


# Main Script
if __name__ == "__main__":
    # Load data
    train_data = np.load('../split_1/train_physionet2012_1.npy', allow_pickle=True)
    validation_data = np.load('../split_1/validation_physionet2012_1.npy', allow_pickle=True)
    test_data = np.load('../split_1/test_physionet2012_1.npy', allow_pickle=True)

    # Create datasets and dataloaders
    train_dataset = EHRDataset(train_data)
    val_dataset = EHRDataset(validation_data)
    test_dataset = EHRDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,collate_fn=custom_collate_fn)

    # Model parameters
    ts_dim = train_data[0]['ts_values'].shape[1]
    static_dim = len(train_data[0]['static'])
    state_dim = 64
    hidden_dim = 128
    num_classes = 2  # Change according to your labels

    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelectiveSSM(ts_dim, static_dim, state_dim, hidden_dim, num_classes).to(device)

    #Xavier Initialization
    model.apply(initialize_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(train_loss)
        val_acc, val_auroc, val_auprc = evaluate_with_metrics(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation AUROC: {val_auroc:.4f}, Validation AUPRC: {val_auprc:.4f}")

    # Test evaluation
    test_acc, test_auroc, test_auprc = evaluate_with_metrics(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}, Test AUROC: {test_auroc:.4f}, Test AUPRC: {test_auprc:.4f}")
