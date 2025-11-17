# Bacformer Comparative Functional Genomics Pipeline

This repository contains a complete end-to-end pipeline for performing **comparative functional genomics** analysis using **Bacformer** (a simulated protein-embedding model).

The goal is to compute **genome-level similarity** using LLM-based protein embeddings, perform **pangenome** and **clustering** analyses, and generate **biological interpretation** using **RAG-based pathway enrichment**.

---

## Directory Structure

```
bacformer_pipeline/
├── data/
│   ├── raw_genomes/
│   │   ├── genome_1.gbk
│   │   ├── genome_1.gff
│   │   ├── ... (50+ Genomes)
│   │   └── genome_N.gff
│   └── reference/
│       ├── go_annotation_db.tsv       # Optional: pathway enrichment DB
│       └── bacformer_rag_index/       # RAG knowledge base files
├── src/
│   ├── run_pipeline.sh                # Orchestrates the workflow (Steps 1–6)
│   ├── extract_proteins.py            # Step 2: Extract FASTA from GBK
│   ├── generate_embeddings.py         # Step 4: Bacformer embeddings
│   ├── compute_similarity.py          # Step 5: Similarity + T-SNE
│   ├── functional_validation.py       # Step 6: Clustering + enrichment
│   └── rag_utils.py                   # RAG helper functions
├── results/
│   ├── all_proteins.fasta
│   ├── all_protein_embeddings.npy
│   ├── protein_ids.csv
│   ├── genome_embeddings.npy
│   ├── genome_similarity_matrix.csv
│   ├── pangenome_roary/
│   │   └── gene_presence_absence.csv
│   └── plots/
│       ├── tsne_genome_clusters.png
│       └── final_umap_clusters.png
├── notebooks/
│   └── exploratory_analysis.ipynb
└── README.md
```

---

## Prerequisites

### 1. Software & Libraries

Install required Python libraries:

```bash
pip install biopython pandas numpy scipy scikit-learn matplotlib tqdm
```

External dependency:

* **Roary** (for pangenome analysis)
  Must be installed and available in your `PATH`.

### 2. Input Data Requirements

All genome files must be placed under:

```
data/raw_genomes/
```

| File Type                  | Requirement          | Source      |
| -------------------------- | -------------------- | ----------- |
| **GenBank (.gbk / .gbff)** | Genome annotation    | NCBI RefSeq |
| **GFF3 (.gff)**            | Annotation for Roary | Same as GBK |

---

## Pipeline Workflow (6 Steps)

The whole pipeline is run through:

```
src/run_pipeline.sh
```

### Steps Overview

| Step  | Script                     | Description                                                | Input         | Output                            |
| ----- | -------------------------- | ---------------------------------------------------------- | ------------- | --------------------------------- |
| **1** | Setup                      | Verify data & directory structure                          | raw genomes   | initialized folders               |
| **2** | `extract_proteins.py`      | Extract protein sequences into master FASTA                | *.gbk         | `results/all_proteins.fasta`      |
| **3** | **Roary**                  | Pangenome analysis → PAV matrix                            | *.gff         | `gene_presence_absence.csv`       |
| **4** | `generate_embeddings.py`   | Bacformer protein embeddings (1024-D)                      | protein FASTA | `all_protein_embeddings.npy`      |
| **5** | `compute_similarity.py`    | Aggregate genome embeddings, compute similarity, T-SNE     | embeddings    | T-SNE plot                        |
| **6** | `functional_validation.py` | Clustering, pangenome enrichment, RAG-based interpretation | all results   | `functional_validation_report.md` |

---

## Usage

### 1. Install Dependencies

```bash
pip install biopython pandas numpy scipy scikit-learn matplotlib tqdm
```

Verify Roary:

```bash
roary -h
```

---

### 2. Download Data

Use included script (optional):

```bash
chmod +x scripts/download_genomes.sh
./scripts/download_genomes.sh
```

Or manually place `.gbk` and `.gff` files into:

```
data/raw_genomes/
```

---

### 3. Run the Full Pipeline

```bash
chmod +x src/run_pipeline.sh
./src/run_pipeline.sh
```

---

### 4. Review Results

All outputs appear in the `results/` directory.

Key final output:

```
results/functional_validation_report.md
```

This report includes:

* Cluster membership
* Pangenome-based signature genes
* Pathway enrichment
* RAG-generated functional summaries

