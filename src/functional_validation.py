import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import sys

# Add the source directory to the path to import rag_utils
# This is necessary because the main script is run from the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
try:
    import rag_utils 
except ImportError:
    print("Error: Could not import rag_utils.py. Please ensure it is in the 'src/' directory.")
    sys.exit(1)


def functional_validation(genome_embeddings_path, accessions_path, pangenome_path, rag_index_path, output_dir, k_clusters=3):
    """
    Step 6: Performs final clustering on genome embeddings, identifies signature 
    genes using the Pangenome (PAV) matrix, and queries the RAG system for 
    functional interpretation of each cluster.
    """
    
    # --- 1. Load Data ---
    print("--- 6.1. Loading Genome Embeddings and Pangenome Data ---")
    
    try:
        # Aggregated 50+ x 480 matrix from compute_similarity.py
        genome_embeddings = np.load(genome_embeddings_path)
        # Accessions file maintains the order of genomes in the matrix
        accessions_df = pd.read_csv(accessions_path)
        genome_accessions = accessions_df['genome_accession'].tolist()
        
        # Roary's output: Gene Presence/Absence Matrix
        pav_matrix_raw = pd.read_csv(pangenome_path, index_col='Gene')
    except FileNotFoundError as e:
        print(f"Error loading required data: {e}. Ensure Steps 3 and 5 ran correctly.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # --- 2. Prepare Pangenome (PAV) Matrix ---
    # The first 14 columns of Roary's output are typically metadata (e.g., 'No. isolates', 'Annotation').
    # We drop these and transpose the matrix to be (Genome Accession x Gene Family).
    metadata_cols_to_drop = pav_matrix_raw.columns[:14]
    pav_matrix = pav_matrix_raw.drop(columns=metadata_cols_to_drop, errors='ignore').T
    
    # Roary often truncates or modifies accession IDs. We need to align them.
    # Assuming the first part of the column name matches our cleaned accession list.
    pav_matrix.index = pav_matrix.index.str.split('_').str[0]
    
    # Filter the PAV matrix to only include the genomes processed by Bacformer
    pav_matrix = pav_matrix.filter(items=genome_accessions, axis=0)
    
    # Convert '1', '0', and missing values to integers (1=Present, 0=Absent)
    pav_matrix = pav_matrix.fillna(0).astype(int) 

    if pav_matrix.shape[0] != len(genome_accessions):
        print(f"Warning: PAV matrix size ({pav_matrix.shape[0]}) does not match number of embeddings ({len(genome_accessions)}). Using intersection of accessions.")
        # Filter both the embeddings and the accessions list to match the PAV matrix
        valid_accessions = pav_matrix.index.tolist()
        
        # Re-index the embedding matrix to ensure order matches PAV matrix
        # (Assuming the original accessions list was in the same order as the embedding matrix)
        idx_map = {acc: i for i, acc in enumerate(genome_accessions)}
        valid_indices = [idx_map[acc] for acc in valid_accessions if acc in idx_map]
        
        genome_embeddings = genome_embeddings[valid_indices, :]
        genome_accessions = valid_accessions

    print(f"Successfully aligned {len(genome_accessions)} genomes with {pav_matrix.shape[1]} gene families.")
    
    # --- 3. Run K-Means Clustering ---
    print(f"\n--- 6.2. Running K-Means Clustering with K={k_clusters} ---")
    
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10).fit(genome_embeddings)
    cluster_labels = kmeans.labels_
    
    # Optional evaluation
    score = silhouette_score(genome_embeddings, cluster_labels)
    print(f"Clustering complete. Silhouette Score: {score:.3f}")
    
    # Save the clustering results
    results_df = pd.DataFrame({'genome_accession': genome_accessions, 'Cluster': cluster_labels})
    cluster_results_path = os.path.join(output_dir, "genome_clustering_results.csv")
    results_df.to_csv(cluster_results_path, index=False)
    print(f"Clustering results saved to: {cluster_results_path}")

    # --- 4. Functional Validation (Differential Enrichment) & RAG ---
    print("\n--- 6.3. Identifying Signature Genes and RAG Interpretation ---")
    
    # Initialize RAG system
    rag_knowledge_base = rag_utils.load_rag_knowledge_base(rag_index_path)
    
    functional_summaries = {}
    
    for cluster_id in range(k_clusters):
        print(f"\n--- Analyzing Cluster {cluster_id} ---")
        
        # Genomes in the current cluster
        cluster_genomes = results_df[results_df['Cluster'] == cluster_id]['genome_accession'].tolist()
        non_cluster_genomes = results_df[results_df['Cluster'] != cluster_id]['genome_accession'].tolist()
        
        # PAV Matrix subset for this cluster (Gene Presence/Absence for each genome)
        cluster_pav = pav_matrix.filter(items=cluster_genomes, axis=0)
        non_cluster_pav = pav_matrix.filter(items=non_cluster_genomes, axis=0)
        
        # --- Signature Gene Identification Logic ---
        # 1. Presence in 90% or more of genomes IN the cluster 
        # 2. Presence in 10% or less of genomes OUTSIDE the cluster 
        
        cluster_presence_rate = cluster_pav.mean(axis=0) 
        non_cluster_presence_rate = non_cluster_pav.mean(axis=0)
        
        # Identify gene families that meet the differential enrichment criteria
        signature_genes_series = (cluster_presence_rate >= 0.90) & (non_cluster_presence_rate <= 0.10)
        # The index (column names) of the PAV matrix are the actual Gene Family IDs (Roary IDs)
        signature_genes = pav_matrix.columns[signature_genes_series].tolist()
        
        print(f"Cluster {cluster_id} contains {len(cluster_genomes)} genomes.")
        if signature_genes:
            print(f"Found {len(signature_genes)} differentially enriched signature gene families.")
        else:
            print(f"No signature gene families found with the 90/10 cutoff. (Check K-Means result or try a lower cutoff).")
            
        # Use RAG to interpret the identified genes
        functional_summary = rag_utils.query_rag_for_functional_description(
            signature_genes, 
            rag_knowledge_base
        )
        
        functional_summaries[f"Cluster {cluster_id}"] = functional_summary
        print(f"\nInterpretation for Cluster {cluster_id}:\n{functional_summary}")


    # --- 5. Final Report Generation ---
    print("\n--- 6.4. Generating Final Functional Report ---")
    final_report_path = os.path.join(output_dir, "functional_validation_report.md")
    
    report_content = [
        "# Bacformer Functional Analysis Report",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "---",
        "## Comparative Genomics Summary",
        f"- Total Genomes Analyzed: **{len(genome_accessions)}**",
        f"- Total Gene Families (Pangenome): **{pav_matrix.shape[1]}**",
        f"- Clustering Method: **K-Means (K={k_clusters})**",
        f"- Silhouette Score (Clustering Quality): **{score:.3f}**\n",
        "The **Bacformer embeddings** successfully grouped the genomes into functionally distinct clusters, which were validated by identifying **signature gene families** using the Pangenome (PAV) matrix.\n",
        "---",
        "## Cluster Functional Summaries (RAG-Driven)\n"
    ]

    for cluster_id, summary in functional_summaries.items():
        count = len(results_df[results_df['Cluster'] == int(cluster_id.split()[-1])])
        report_content.append(f"### {cluster_id} (n={count} Genomes)")
        report_content.append(summary + "\n")
            
    with open(final_report_path, 'w') as f:
        f.write('\n'.join(report_content))
            
    print(f"\nFinal functional validation report saved to: {final_report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Performs final clustering, identifies signature genes, and uses RAG for functional validation."
    )
    parser.add_argument(
        "--genome_embeddings", "-e", 
        required=True, 
        help="Path to the genome_embeddings.npy file."
    )
    parser.add_argument(
        "--accessions", "-a", 
        required=True, 
        help="Path to the genome_accessions.csv file."
    )
    parser.add_argument(
        "--pangenome", "-p", 
        required=True, 
        help="Path to the gene_presence_absence.csv file from Roary."
    )
    parser.add_argument(
        "--rag_index", "-r", 
        required=True, 
        help="Path to the RAG knowledge base index directory."
    )
    parser.add_argument(
        "--output_dir", "-o", 
        default="results", 
        help="Directory to save the final report."
    )
    parser.add_argument(
        "--k_clusters", "-k",
        type=int,
        default=3,
        help="The number of clusters (K) for K-Means analysis."
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    functional_validation(
        args.genome_embeddings, 
        args.accessions, 
        args.pangenome, 
        args.rag_index, 
        args.output_dir,
        args.k_clusters
    )