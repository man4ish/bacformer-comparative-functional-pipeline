import argparse
import os
import glob
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def extract_proteins(input_dir, output_fasta):
    """
    Scans an input directory for .gbk or .gbff files, extracts all protein
    sequences (translations) from CDS features, and compiles them into a single
    FASTA file.
    """
    
    # Use glob to find all GenBank files in the input directory
    # Searches for common extensions: .gbk, .gbff, .genbank
    genbank_files = glob.glob(os.path.join(input_dir, '*.gbk')) + \
                    glob.glob(os.path.join(input_dir, '*.gbff'))
    
    if not genbank_files:
        print(f"Error: No .gbk or .gbff files found in {input_dir}")
        return

    all_protein_records = []
    total_proteins = 0
    
    # Process each GenBank file
    for gbk_path in genbank_files:
        print(f"Processing: {os.path.basename(gbk_path)}")
        
        try:
            for record in SeqIO.parse(gbk_path, "genbank"):
                genome_accession = record.id.split('.')[0] # Use accession up to first dot
                
                for feature in record.features:
                    if feature.type == "CDS" and "translation" in feature.qualifiers:
                        protein_seq = feature.qualifiers["translation"][0]
                        
                        # Generate a clean ID for the FASTA header
                        # Prioritize Locus Tag, then fall back to protein_id, then just use a counter
                        protein_id = feature.qualifiers.get("locus_tag", feature.qualifiers.get("protein_id", [f"protein_{total_proteins}"]))[0]

                        # Create a Biopython SeqRecord for proper FASTA output
                        seq_record = SeqRecord(
                            Seq(protein_seq),
                            id=f"{genome_accession}|{protein_id}",
                            description=f"Protein from genome {genome_accession}"
                        )
                        all_protein_records.append(seq_record)
                        total_proteins += 1
                        
        except Exception as e:
            print(f"Warning: Could not parse file {os.path.basename(gbk_path)}. Error: {e}")
            continue

    print(f"\nSuccessfully extracted {total_proteins} proteins from {len(genbank_files)} genomes.")

    # Save to a single output FASTA file
    with open(output_fasta, "w") as f:
        SeqIO.write(all_protein_records, f, "fasta")

    print(f"All protein sequences saved to: {output_fasta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts all protein sequences from a directory of GenBank files into a single FASTA file."
    )
    parser.add_argument(
        "--input_dir", "-i", 
        required=True, 
        help="Path to the directory containing all GenBank (.gbk/.gbff) files."
    )
    parser.add_argument(
        "--output_fasta", "-o", 
        default="results/all_proteins.fasta", 
        help="Path to the output FASTA file for all protein sequences."
    )
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_fasta), exist_ok=True)
    
    extract_proteins(args.input_dir, args.output_fasta)