import argparse
from Bio import SeqIO

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Extract protein sequences from a GenBank file.")
parser.add_argument("--input", "-i", required=True, help="Path to the input GenBank (.gbk/.gbff) file")
parser.add_argument("--output", "-o", default="proteins.txt", help="Output text file for protein sequences")
args = parser.parse_args()

genbank_file = args.input
output_file = args.output

protein_sequences = []

# Parse GenBank file
for record in SeqIO.parse(genbank_file, "genbank"):
    for feature in record.features:
        if feature.type == "CDS" and "translation" in feature.qualifiers:
            protein_sequences.append(feature.qualifiers["translation"][0])

print(f"Extracted {len(protein_sequences)} proteins")

# Save to output file
with open(output_file, "w") as f:
    for seq in protein_sequences:
        f.write(seq + "\n")

print(f"Protein sequences saved to {output_file}")
