from transformers.models.mamba.configuration_mamba import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import numpy as np
from collections import Counter

myfile=open("results.txt", "w")

# Custom PyTorch Dataset
class EHRDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ts_values = torch.tensor(item['ts_values'], dtype=torch.float32)
        static = torch.tensor(item['static'], dtype=torch.float32)
        label = torch.tensor(item['labels'], dtype=torch.long)
        return ts_values, static, label


# Custom collate_fn
def custom_collate_fn(batch):
    ts_values = [item[0] for item in batch]
    static_features = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])

    ts_values_padded = pad_sequence(ts_values, batch_first=True)
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)

    return ts_values_padded, static_features, labels, lengths


# Cuantización de tensores con ajuste de rango
def quantize_tensor(tensor, num_bins):
    """Convierte valores continuos en índices discretos después de normalización."""
    min_val, max_val = tensor.min(), tensor.max()

    # Ajustar los valores al rango positivo [0, max_val - min_val]
    tensor_normalized = (tensor - min_val) / (max_val - min_val)

    # Crear bins en el rango [0, 1] y cuantizar
    bins = torch.linspace(0, 1, steps=num_bins + 1, device=tensor.device)
    quantized = torch.bucketize(tensor_normalized, bins) - 1  # Ajustar índice para empezar en 0
    quantized = quantized.clamp(0, num_bins - 1)  # Asegurar que los índices estén dentro del rango válido

    # Reducir a 2D seleccionando el índice más significativo
    return quantized.argmax(-1)


# Training Function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_bins = 128  # Número de bins para la cuantización
    for ts_values, static_features, labels, lengths in train_loader:
        ts_values, static_features, labels, lengths = (
            ts_values.to(device),
            static_features.to(device),
            labels.to(device),
            lengths.to(device),
        )

        # Cuantiza los valores continuos de ts_values
        ts_values_quantized = quantize_tensor(ts_values, num_bins)

        # Validar que los índices están en rango
        if (ts_values_quantized >= num_bins).any() or (ts_values_quantized < 0).any():
            raise ValueError("Los valores cuantizados están fuera del rango esperado.")

        optimizer.zero_grad()

        # Forward pass
        attention_mask = (ts_values_quantized > 0).float()  # Crear máscara de atención 2D
        outputs = model(input_ids=ts_values_quantized, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, -1, :]  # Use the last token's logits for classification
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Evaluation Function
def evaluate_with_metrics(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    num_bins = 50  # Número de bins para la cuantización
    with torch.no_grad():
        for ts_values, static_features, labels, lengths in loader:
            ts_values, static_features, labels, lengths = (
                ts_values.to(device),
                static_features.to(device),
                labels.to(device),
                lengths.to(device),
            )

            # Cuantiza los valores continuos de ts_values
            ts_values_quantized = quantize_tensor(ts_values, num_bins)

            outputs = model(input_ids=ts_values_quantized, attention_mask=(ts_values_quantized > 0))
            logits = outputs.logits
            logits = logits[:, -1, :]  # Use the last token's logits for classification
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    return acc, auroc, auprc


# Main Script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_data = np.load('../P12data/split_1/split_1/train_physionet2012_1.npy', allow_pickle=True)
    validation_data = np.load('../P12data/split_1/split_1/validation_physionet2012_1.npy', allow_pickle=True)
    test_data = np.load('../P12data/split_1/split_1/test_physionet2012_1.npy', allow_pickle=True)

    # Create datasets and dataloaders
    train_dataset = EHRDataset(train_data)
    val_dataset = EHRDataset(validation_data)
    test_dataset = EHRDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=custom_collate_fn,num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, collate_fn=custom_collate_fn,num_workers=4, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True, shuffle=False)

    # Initialize MambaForCausalLM from scratch
    config = MambaConfig(
        vocab_size=5000,  # Número de bins usados en la cuantización
        n_positions=512,  # Máxima longitud de secuencia
        hidden_size=256,  # Dimensión oculta del modelo
        num_hidden_layers=8,  # Número de capas
        num_attention_heads=8,  # Número de cabezas de atención
        intermediate_size=512,  # Tamaño de la capa intermedia
        hidden_dropout_prob=0.2,  # Dropout en las capas ocultas
        attention_probs_dropout_prob=0.2,
    )

    model = MambaForCausalLM(config=config)
    model.to(device)
    # Ajusta la última capa para tener 2 salidas
    model.lm_head = nn.Linear(config.hidden_size, 2).to(device)

    class_counts = Counter([item['labels'] for item in train_data])
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / count for count in class_counts.values()]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Loss, optimizer, and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        print("Starting training in epoch:", epoch)
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_acc, val_auroc, val_auprc = evaluate_with_metrics(model, val_loader, device)
        myfile.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation AUROC: {val_auroc:.4f}, Validation AUPRC: {val_auprc:.4f}")
        myfile.write("\n")
        scheduler.step()

    # Test evaluation
    test_acc, test_auroc, test_auprc = evaluate_with_metrics(model, test_loader, device)
    myfile.write(f"Test Accuracy: {test_acc:.4f}, Test AUROC: {test_auroc:.4f}, Test AUPRC: {test_auprc:.4f}")
    myfile.write("\n")
