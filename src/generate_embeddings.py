import argparse
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from transformers import AutoModel

# Assuming the bacformer library is installed and available
from bacformer.pp import protein_seqs_to_bacformer_inputs 

def generate_embeddings(input_fasta, output_embeddings, output_ids):
    """
    Generates 480D Bacformer embeddings for every protein sequence in the input FASTA file.
    Saves embeddings to a .npy file and corresponding IDs to a .csv file.
    """
    
    print("--- 4.1. Loading Protein Data ---")
    
    # Correctly parse the multi-FASTA file
    protein_records = list(SeqIO.parse(input_fasta, "fasta"))
    if not protein_records:
        print(f"Error: No protein records found in {input_fasta}. Exiting.")
        return

    # Extract sequences and corresponding IDs
    sequences = [str(r.seq) for r in protein_records]
    metadata = {
        "protein_id": [r.id for r in protein_records],
        # The ID format is "Accession|LocusTag" from extract_proteins.py
        "genome_accession": [r.id.split('|')[0] for r in protein_records]
    }
    
    total_proteins = len(sequences)
    print(f"Loaded {total_proteins} total protein sequences.")

    # --- Setup Model and Device ---
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "macwiatrak/bacformer-masked-MAG"
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading Bacformer model: {e}")
        return

    # --- Generate Embeddings ---
    print("\n--- 4.2. Generating Embeddings in Batches ---")
    
    # Note: max_n_proteins=6000 is usually too large for GPU VRAM. Reducing batch size.
    BATCH_SIZE = 128 
    
    # bacformer.pp.protein_seqs_to_bacformer_inputs handles tokenization and batching
    # for the Bacformer model structure.
    inputs = protein_seqs_to_bacformer_inputs(
        sequences,
        device=device,
        batch_size=BATCH_SIZE,
        # Max proteins refers to the maximum number of sequences to process, 
        # which should be the total number of proteins we found.
        max_n_proteins=total_proteins 
    )

    all_embeddings = []

    # Run model to get embeddings in a controlled loop
    with torch.no_grad():
        for i in range(0, total_proteins, BATCH_SIZE):
            # Extract the current batch inputs from the prepared dict
            batch_inputs = {
                'input_ids': inputs['input_ids'][i:i+BATCH_SIZE].to(device),
                'attention_mask': inputs['attention_mask'][i:i+BATCH_SIZE].to(device)
            }
            
            # Skip if input is empty (shouldn't happen but good practice)
            if batch_inputs['input_ids'].size(0) == 0:
                continue

            outputs = model(**batch_inputs, return_dict=True)
            
            # The protein embedding is the CLS token output (first position, dim=1)
            # which is 480D (or similar, depending on model config).
            protein_embeddings = outputs.last_hidden_state[:, 0, :]
            
            all_embeddings.append(protein_embeddings.cpu().numpy())
            print(f"Processed batch {i // BATCH_SIZE + 1}. Total embedded: {i + len(protein_embeddings)}")

    # Combine all batch embeddings into a single NumPy array
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    
    print("\n--- 4.3. Saving Results ---")
    
    # Final shape check (should be N proteins x 480 dimensions)
    print(f"Final Embeddings Shape: {final_embeddings.shape}") 
    
    # 1. Save embeddings to .npy file
    np.save(output_embeddings, final_embeddings)
    print(f"All protein embeddings saved to '{output_embeddings}'")

    # 2. Save IDs and metadata to a CSV file
    id_df = pd.DataFrame(metadata)
    id_df.to_csv(output_ids, index=False)
    print(f"Protein metadata (ID, Accession) saved to '{output_ids}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Bacformer embeddings from a multi-FASTA file of protein sequences."
    )
    parser.add_argument(
        "--input_fasta", "-i", 
        required=True, 
        help="Path to the input multi-FASTA file (e.g., results/all_proteins.fasta)."
    )
    parser.add_argument(
        "--output_embeddings", "-e", 
        default="results/all_protein_embeddings.npy", 
        help="Output .npy file for all protein embeddings."
    )
    parser.add_argument(
        "--output_ids", "-d", 
        default="results/protein_ids.csv", 
        help="Output CSV file containing the corresponding protein IDs and genome accessions."
    )
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_embeddings), exist_ok=True)
    
    generate_embeddings(args.input_fasta, args.output_embeddings, args.output_ids)