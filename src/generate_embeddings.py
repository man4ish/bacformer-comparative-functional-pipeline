import argparse
import torch
import numpy as np
from transformers import AutoModel
from bacformer.pp import protein_seqs_to_bacformer_inputs

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Generate Bacformer embeddings from protein sequences")
parser.add_argument("--input", "-i", required=True, help="Path to protein sequences file (one per line)")
parser.add_argument("--output", "-o", default="genome_embedding.npy", help="Output .npy file for genome embedding")
args = parser.parse_args()

input_file = args.input
output_file = args.output

# Load protein sequences
with open(input_file) as f:
    protein_sequences = [line.strip() for line in f if line.strip()]

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load Bacformer model
model_name = "macwiatrak/bacformer-masked-MAG"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()
print("Model loaded successfully!")

# Convert protein sequences to Bacformer inputs
inputs = protein_seqs_to_bacformer_inputs(
    protein_sequences,
    device=device,
    batch_size=128,
    max_n_proteins=6000
)

# Run model to get embeddings
with torch.no_grad():
    outputs = model(**inputs, return_dict=True)

# Genome embedding: mean of all protein embeddings
genome_embedding = outputs.last_hidden_state.mean(dim=1)

print("Last hidden state shape:", outputs["last_hidden_state"].shape)
print("Genome embedding shape:", genome_embedding.shape)
print("First 5 values:", genome_embedding[0][:5])

# Save the genome embedding
np.save(output_file, genome_embedding.cpu().numpy())
print(f"Genome embedding saved to '{output_file}'")
