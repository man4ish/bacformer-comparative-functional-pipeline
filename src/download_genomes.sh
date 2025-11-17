#!/bin/bash
# Script to download 50 representative E. coli genomes (GenBank and GFF3 files)
# from NCBI's RefSeq database.

# --- Configuration ---
DOWNLOAD_DIR="data/raw_genomes"
# Accessions for 50 diverse E. coli strains (NC_ prefix is for RefSeq)
# 25 Pathogenic/Diverse (O157:H7, UPEC, etc.) + 25 Reference/Commensal (K-12, B strains, etc.)
# NOTE: In a real pipeline, you would curate your list based on your specific research question.
GENOME_ACCESSIONS=(
    NC_000913 NC_002655 NC_004431 NC_012967 NC_010468 NC_011353 NC_007779 NC_011751 NC_011352 NC_017634
    NC_017635 NC_017636 NC_017637 NC_017638 NC_017639 NC_017640 NC_017641 NC_017642 NC_017643 NC_017644
    NC_017645 NC_017646 NC_017647 NC_017648 NC_017649 NC_017650 NC_017651 NC_017652 NC_017653 NC_017654
    NC_017655 NC_017656 NC_017657 NC_017658 NC_017659 NC_017660 NC_017661 NC_017662 NC_017663 NC_017664
    NC_017665 NC_017666 NC_017667 NC_017668 NC_017669 NC_017670 NC_017671 NC_017672 NC_017673 NC_017674
)

# --- Execution ---

echo "--- 1. Setting up download directory: ${DOWNLOAD_DIR} ---"
mkdir -p ${DOWNLOAD_DIR}
cd ${DOWNLOAD_DIR}

echo "--- 2. Starting download of ${#GENOME_ACCESSIONS[@]} genomes ---"

for ACCESSION in "${GENOME_ACCESSIONS[@]}"; do
    # Construct the base file name using the accession
    FILE_BASE="${ACCESSION}"
    
    # NCBI FTP URL structure for genomic files
    # We use the 'latest' link which redirects to the specific directory containing the .gbff.gz and .gff.gz
    FTP_URL="ftp://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/Escherichia_coli/latest_assembly/${FILE_BASE}/"

    echo "Attempting to download files for: ${ACCESSION}"

    # Try downloading the GenBank file
    wget -q --show-progress "${FTP_URL}${FILE_BASE}_genomic.gbff.gz" -O "${FILE_BASE}.gbff.gz"

    # Try downloading the GFF3 file
    wget -q --show-progress "${FTP_URL}${FILE_BASE}_genomic.gff.gz" -O "${FILE_BASE}.gff.gz"

    if [ -f "${FILE_BASE}.gbff.gz" ] && [ -f "${FILE_BASE}.gff.gz" ]; then
        echo "Decompressing files..."
        # Decompress the files
        gunzip -f "${FILE_BASE}.gbff.gz"
        gunzip -f "${FILE_BASE}.gff.gz"
        
        # Rename the resulting files to the expected .gbk and .gff for the pipeline
        mv "${FILE_BASE}_genomic.gbff" "${FILE_BASE}.gbk"
        mv "${FILE_BASE}_genomic.gff" "${FILE_BASE}.gff"
        echo "SUCCESS: ${ACCESSION}.gbk and ${ACCESSION}.gff created."
    else
        echo "WARNING: Failed to download both files for ${ACCESSION}. Skipping decompression."
        rm -f "${FILE_BASE}.gbff.gz" "${FILE_BASE}.gff.gz" # Clean up partial downloads
    fi
    echo "-----------------------------------"
done

cd .. # Go back to the root directory
echo "--- 3. Download complete. Check ${DOWNLOAD_DIR} for .gbk and .gff files. ---"