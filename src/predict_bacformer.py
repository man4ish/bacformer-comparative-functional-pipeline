import torch
from bacformer.pp import protein_seqs_to_bacformer_inputs
from finetune_bacformer_classification import BacformerClassifier, AutoModel, device, le  # reuse your label encoder

# ------------------------------
# 1️⃣ Parameters
# ------------------------------
MODEL_NAME = "macwiatrak/bacformer-masked-MAG"
FINETUNED_MODEL = "bacformer_finetuned_classification.pt"

# ------------------------------
# 2️⃣ Load Bacformer base model
# ------------------------------
bacformer_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
bacformer_model.eval()

# ------------------------------
# 3️⃣ Load classifier head
# ------------------------------
num_labels = len(le.classes_)
model = BacformerClassifier(bacformer_model, num_labels).to(device)
model.load_state_dict(torch.load(FINETUNED_MODEL, map_location=device))
model.eval()

# ------------------------------
# 4️⃣ Function to predict single sequence
# ------------------------------
def predict_sequence(seq):
    inputs = protein_seqs_to_bacformer_inputs([seq], device=device, batch_size=1)
    with torch.no_grad():
        logits = model(inputs)
        pred_idx = torch.argmax(logits, dim=1).item()
        label = le.inverse_transform([pred_idx])[0]
    return label

# ------------------------------
# 5️⃣ Predict multiple sequences from a file
# ------------------------------
def predict_file(file_path):
    with open(file_path) as f:
        sequences = [line.strip() for line in f if line.strip()]

    predictions = []
    for seq in sequences:
        label = predict_sequence(seq)
        predictions.append((seq, label))
    return predictions

# ------------------------------
# 6️⃣ Example usage
# ------------------------------
if __name__ == "__main__":
    # Single sequence
    seq = "MKLIVVLLVTLVLCQGYT"
    print(f"Sequence: {seq} -> Predicted label: {predict_sequence(seq)}")

    # Multiple sequences from file
    pred_file = "data/test_proteins.txt"  # one sequence per line
    results = predict_file(pred_file)
    for seq, label in results:
        print(f"Sequence: {seq} -> Predicted label: {label}")
