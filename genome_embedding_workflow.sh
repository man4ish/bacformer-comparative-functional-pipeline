#!/bin/sh
set -e  # Exit immediately if a command fails

# --- Create output directories if they don't exist ---
mkdir -p out gbk_files

# --- Download GenBank files from NCBI ---
echo "Downloading Genome 1 (Bacillus subtilis)..."
wget -O gbk_files/GCF_000009045.1_ASM904v1_genomic.gbff.gz \
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/009/045/GCF_000009045.1_ASM904v1/GCF_000009045.1_ASM904v1_genomic.gbff.gz

echo "Downloading Genome 2 (E. coli)..."
wget -O gbk_files/GCF_000005845.2_ASM584v2_genomic.gbff.gz \
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.gbff.gz

# --- Decompress downloaded files ---
echo "Decompressing Genome 1..."
gunzip -c gbk_files/GCF_000009045.1_ASM904v1_genomic.gbff.gz > gbk_files/GCF_000009045.1_ASM904v1_genomic.gbff

echo "Decompressing Genome 2..."
gunzip -c gbk_files/GCF_000005845.2_ASM584v2_genomic.gbff.gz > gbk_files/GCF_000005845.2_ASM584v2_genomic.gbff

# --- Extract proteins from GenBank files ---
echo "Extracting proteins from Genome 1..."
python src/extract_proteins.py --input gbk_files/GCF_000009045.1_ASM904v1_genomic.gbff --output out/proteins_genome2.txt

echo "Extracting proteins from Genome 2..."
python src/extract_proteins.py --input gbk_files/GCF_000005845.2_ASM584v2_genomic.gbff --output out/proteins_genome1.txt

# --- Generate Bacformer embeddings ---
echo "Generating embeddings for Genome 1..."
python src/generate_embeddings.py --input out/proteins_genome1.txt --output out/genome1_embedding.npy

echo "Generating embeddings for Genome 2..."
python src/generate_embeddings.py --input out/proteins_genome2.txt --output out/genome2_embedding.npy

# --- Compute pairwise similarity ---
echo "Computing pairwise similarity..."
python src/compare_embeddings.py --embeddings out/genome1_embedding.npy out/genome2_embedding.npy

echo "All genomes processed. Embeddings and similarity matrix are ready."


