import argparse
import numpy as np
from scipy.spatial.distance import cosine

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Compute pairwise cosine similarity between genome embeddings")
parser.add_argument("--embeddings", "-e", nargs="+", required=True,
                    help="List of genome embedding .npy files")
args = parser.parse_args()

embedding_files = args.embeddings
embeddings = []

# Load all embeddings
for f in embedding_files:
    emb = np.load(f)
    embeddings.append(emb.flatten())  # flatten if shape is (1,480)

n = len(embeddings)
print(f"Computing pairwise similarity for {n} genomes...")

# Compute pairwise cosine similarity
similarity_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        similarity_matrix[i, j] = 1 - cosine(embeddings[i], embeddings[j])

# Print similarity matrix
print("\nPairwise Cosine Similarity Matrix:")
for i in range(n):
    row = " ".join(f"{similarity_matrix[i,j]:.4f}" for j in range(n))
    print(row)

# Save matrix to file
np.savetxt("out/similarity_matrix.txt", similarity_matrix, fmt="%.4f")
print("\nSimilarity matrix saved to 'similarity_matrix.txt'")
