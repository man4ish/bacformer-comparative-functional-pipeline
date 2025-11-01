import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bacformer.pp import protein_seqs_to_bacformer_inputs
from transformers import AutoModel
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# 1️⃣ Parameters
# ------------------------------
TRAIN_FILE = "data/train_labels.csv"
VAL_FILE = "data/val_labels.csv"
BATCH_SIZE = 2   # adjust based on memory
EPOCHS = 3
LEARNING_RATE = 2e-5
MODEL_NAME = "macwiatrak/bacformer-masked-MAG"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------------------
# 2️⃣ Load dataset
# ------------------------------
train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

# Encode labels
le = LabelEncoder()
train_df["labels"] = le.fit_transform(train_df["label"])
val_df["labels"] = le.transform(val_df["label"])

# ------------------------------
# 3️⃣ PyTorch Dataset
# ------------------------------
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = ProteinDataset(list(train_df.sequence), list(train_df.labels))
val_dataset = ProteinDataset(list(val_df.sequence), list(val_df.labels))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ------------------------------
# 4️⃣ Load Bacformer model
# ------------------------------
bacformer_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
bacformer_model.eval()  # freeze weights during initial training

# ------------------------------
# 5️⃣ Classification head
# ------------------------------
class BacformerClassifier(nn.Module):
    def __init__(self, bacformer_model, num_labels):
        super().__init__()
        self.bacformer = bacformer_model
        self.classifier = nn.Linear(480, num_labels)  # 480-dim genome embedding

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.bacformer(**inputs, return_dict=True)
        genome_emb = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(genome_emb)
        return logits

num_labels = len(le.classes_)
model = BacformerClassifier(bacformer_model, num_labels).to(device)

# ------------------------------
# 6️⃣ Optimizer and loss
# ------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# ------------------------------
# 7️⃣ Training loop (fixed batching)
# ------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        sequences = batch["sequence"]
        labels = batch["label"].to(device)

        # Process each sequence individually and collect embeddings
        all_logits = []
        for seq in sequences:
            # Convert single sequence to Bacformer input
            inputs = protein_seqs_to_bacformer_inputs([seq], device=device, batch_size=1)
            logit = model(inputs)
            all_logits.append(logit)

        # Concatenate logits to form batch
        logits = torch.cat(all_logits, dim=0)  # shape = (batch_size, num_labels)

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")

    # ------------------------------
    # Validation
    # ------------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            sequences = batch["sequence"]
            labels = batch["label"].to(device)

            all_logits = []
            for seq in sequences:
                inputs = protein_seqs_to_bacformer_inputs([seq], device=device, batch_size=1)
                logit = model(inputs)
                all_logits.append(logit)

            logits = torch.cat(all_logits, dim=0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Validation Accuracy: {val_acc:.4f}")

# ------------------------------
# 8️⃣ Save model
# ------------------------------
torch.save(model.state_dict(), "bacformer_finetuned_classification.pt")
print("Fine-tuned model saved as 'bacformer_finetuned_classification.pt'")
