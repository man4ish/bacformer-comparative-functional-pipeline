
# Bacformer Genome Embedding and Comparative Analysis Pipeline

This repository provides a **pipeline to extract protein sequences from bacterial genomes, generate Bacformer embeddings, and perform comparative analysis** using cosine similarity. It is designed to work with GenBank (`.gbk` / `.gbff`) files and is compatible with macOS, Linux, or KBase environments.

---

## **Features**

- Extract coding sequences (CDS) from GenBank files and convert them to protein sequences.
- Generate **Bacformer embeddings** (contextualized protein language model) for each genome.
- Compute **pairwise cosine similarity** between genomes for functional comparison.
- Save embeddings as `.npy` files for downstream analysis.
- Fully parameterized for multiple genomes.

---

## **Requirements**

See [`requirements.txt`](requirements.txt) for all dependencies:

```txt
biopython>=1.80
numpy>=1.24
torch>=2.1
torchvision>=0.16
torchaudio>=2.1
transformers>=4.40
bacformer>=0.0.5
huggingface_hub>=0.16
scipy>=1.12
````

Install all dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## **Usage**

### **1. Extract protein sequences from GenBank files**

```bash
python src/extract_proteins.py --input genome1.gbk --output proteins_genome1.txt
python src/extract_proteins.py --input genome2.gbk --output proteins_genome2.txt
```

* Produces a `.txt` file with one protein sequence per line.

---

### **2. Generate Bacformer embeddings**

```bash
python src/generate_embeddings.py --input proteins_genome1.txt --output genome1_embedding.npy
python src/generate_embeddings.py --input proteins_genome2.txt --output genome2_embedding.npy
```

* Produces a `.npy` file with a 480-dimensional genome embedding.

---

### **3. Compute pairwise similarity**

```bash
python src/compare_embeddings.py --embeddings genome1_embedding.npy genome2_embedding.npy
```

* Prints a **cosine similarity matrix** to the console.
* Saves the similarity matrix to `similarity_matrix.txt`.

---

### **4. End-to-end script**

You can use the included shell script to process multiple genomes automatically:

```bash
chmod +x run_genomes.sh
./run_genomes.sh
```

* Extracts proteins, generates embeddings, and computes pairwise similarity for all genomes in the script.

---

## **Output**

* `proteins_genomeX.txt` — extracted protein sequences.
* `genomeX_embedding.npy` — Bacformer embedding for each genome.
* `similarity_matrix.txt` — pairwise cosine similarity for all processed genomes.

---

## **Notes**

* Works on **CPU or Apple M-series GPU (MPS)**; for large genomes or batch processing, NVIDIA GPU is recommended.
* Bacformer model is **public**: `macwiatrak/bacformer-masked-MAG`.
* Optional: install [faESM](https://github.com/pengzhangzhi/faplm) or `flash-attn` for faster embeddings.

Here’s an updated **Fine-Tuning section** you can add to your current README, integrating your classification fine-tuning code:

---

### **5. Fine-Tuning Bacformer for Supervised Classification**

You can fine-tune Bacformer for a **protein-level classification task** (e.g., antibiotic resistance prediction). This requires labeled sequences in CSV format.

#### **Prepare your dataset**

* Training CSV (`train_labels.csv`) and validation CSV (`val_labels.csv`) should have two columns:

| sequence           | label        |
| ------------------ | ------------ |
| MKLIVVLLVTLVLCQGYT | resistant    |
| MAKLTIVLTLVLCQGYT  | nonresistant |

* Ensure **one protein sequence per row** and labels match your classification categories.

---

#### **Run fine-tuning**

```bash
python src/finetune_bacformer_classification.py \
    --train_csv data/train_labels.csv \
    --val_csv data/val_labels.csv \
    --epochs 10 \
    --batch_size 8 \
    --output bacformer_finetuned_classification.pt
```

* Adjust `epochs` and `batch_size` depending on your dataset and device.
* The script will save the fine-tuned model as `bacformer_finetuned_classification.pt`.

---

#### **Inference with Fine-Tuned Model**

Once fine-tuned, you can predict labels for new sequences:

```bash
python src/predict_bacformer.py
```

* The script reads a file `data/test_proteins.txt` (one sequence per line) and outputs predicted labels.
* You can also provide a single sequence in the script for quick testing.

---

#### **Notes**

* **Device support:** Works on CPU or Apple M-series GPU (MPS). For large datasets, NVIDIA GPU is recommended.
* **Weights warning:**

  ```
  Some weights of EsmModel were not initialized from the model checkpoint...
  ```

  This is normal — newly added classification layers are trained during fine-tuning.
* **Accuracy tips:**

  * More sequences per class improve performance.
  * Consider unfreezing some Bacformer layers for better downstream adaptation.

---

## **References**

* Wiatrak et al., *“A contextualised protein language model reveals the functional syntax of bacterial evolution”*, 2025.
* Bacformer Hugging Face model: [https://huggingface.co/macwiatrak/bacformer-masked-MAG](https://huggingface.co/macwiatrak/bacformer-masked-MAG)
* GitHub repo: [https://github.com/macwiatrak/Bacformer](https://github.com/macwiatrak/Bacformer)


