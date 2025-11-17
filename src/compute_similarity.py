import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def aggregate_and_cluster(embeddings_path, ids_path, output_dir):
    """
    1. Loads protein embeddings and metadata.
    2. Aggregates protein embeddings by genome (calculates the mean).
    3. Computes the pairwise cosine similarity matrix between genomes.
    4. Performs T-SNE visualization on the genome embeddings.
    """
    
    # --- 1. Load Data ---
    print("--- 5.1. Loading Protein Embeddings and Metadata ---")
    
    # Load the massive N_proteins x 480 embedding matrix
    try:
        protein_embeddings = np.load(embeddings_path)
    except FileNotFoundError:
        print(f"Error: Embeddings file not found at {embeddings_path}. Run Step 4 first.")
        return
        
    # Load the corresponding IDs/metadata
    try:
        metadata_df = pd.read_csv(ids_path)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {ids_path}. Run Step 4 first.")
        return

    # Check for shape consistency
    if len(protein_embeddings) != len(metadata_df):
        print("Error: Number of embeddings does not match number of protein IDs. Check extraction/embedding scripts.")
        return
        
    print(f"Loaded {len(protein_embeddings)} protein embeddings.")

    # --- 2. Aggregate Protein Embeddings to Genome Embeddings ---
    print("\n--- 5.2. Aggregating Protein Embeddings by Genome (Mean) ---")
    
    # Create a DataFrame combining metadata and embeddings
    # This allows us to use pandas groupby functionality for efficient aggregation
    embedding_df = pd.DataFrame(protein_embeddings)
    embedding_df['genome_accession'] = metadata_df['genome_accession']
    
    # Group by genome accession and compute the mean of all protein vectors
    # The result is one 480D vector (row) per unique genome.
    genome_embeddings_df = embedding_df.groupby('genome_accession').mean()
    genome_accessions = genome_embeddings_df.index.tolist()
    genome_embeddings = genome_embeddings_df.values
    
    n_genomes = len(genome_accessions)
    print(f"Successfully aggregated embeddings for {n_genomes} unique genomes.")
    print(f"Genome Embeddings Shape: {genome_embeddings.shape}")
    
    # --- 3. Compute Pairwise Cosine Similarity ---
    print("\n--- 5.3. Computing Pairwise Cosine Similarity Matrix ---")
    
    similarity_matrix = np.zeros((n_genomes, n_genomes))
    
    for i in range(n_genomes):
        for j in range(n_genomes):
            # Cosine distance is the angle between vectors; 1 - distance gives similarity
            similarity_matrix[i, j] = 1 - cosine(genome_embeddings[i], genome_embeddings[j])

    # Save matrix to file
    matrix_output_path = os.path.join(output_dir, "genome_similarity_matrix.csv")
    np.savetxt(matrix_output_path, similarity_matrix, delimiter=",", fmt="%.4f")
    print(f"Similarity matrix saved to '{matrix_output_path}'")
    
    # Save the aggregated genome embeddings and their order
    np.save(os.path.join(output_dir, "genome_embeddings.npy"), genome_embeddings)
    pd.DataFrame({'genome_accession': genome_accessions}).to_csv(
        os.path.join(output_dir, "genome_accessions.csv"), index=False
    )
    
    print("Aggregated genome embeddings saved for clustering/visualization.")
    
    
    # --- 4. T-SNE Visualization for Initial Clustering ---
    print("\n--- 5.4. Performing T-SNE Visualization ---")
    
    # Reduce dimensions from 480 to 2 for plotting
    tsne = TSNE(n_components=2, random_state=42, perplexity=15, metric='cosine', init='random', n_jobs=-1)
    
    # NOTE: The similarity matrix is usually used for plotting, but for simplicity
    # and consistency with modern methods, we apply T-SNE directly to the embeddings.
    tsne_results = tsne.fit_transform(genome_embeddings)
    
    # Plotting the T-SNE results
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7, edgecolors='w', linewidths=0.5)
    plt.title('Bacformer Genome Embeddings T-SNE Visualization')
    plt.xlabel('T-SNE Component 1')
    plt.ylabel('T-SNE Component 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_path = os.path.join(output_dir, "tsne_genome_clusters.png")
    plt.savefig(plot_path)
    print(f"T-SNE visualization saved to '{plot_path}' .")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregates protein embeddings into genome embeddings, computes pairwise similarity, and generates a T-SNE plot."
    )
    parser.add_argument(
        "--embeddings", "-e", 
        required=True, 
        help="Path to the all_protein_embeddings.npy file."
    )
    parser.add_argument(
        "--protein_ids", "-d", 
        required=True, 
        help="Path to the protein_ids.csv file (metadata)."
    )
    parser.add_argument(
        "--output_dir", "-o", 
        default="results", 
        help="Directory to save the similarity matrix and plots."
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    aggregate_and_cluster(args.embeddings, args.protein_ids, args.output_dir)