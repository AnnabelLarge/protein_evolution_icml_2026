# Code for "Nested birth-death processes are competitive with neural networks as protein evolution models", ICML 2026.

This repo contains code to preprocess Pfam v36.0 into pairwise alignments, train and evaluate pairHMM and neural models of protein sequence evolution, and reproduce all reported log-likelihoods.
We also include details about the three train/dev/test partitions, as well as log-likelihood metrics for all models trained on the three training replicates.  

The bioRxiv preprint can be found here: [https://doi.org/10.64898/2026.02.02.702952](https://doi.org/10.64898/2026.02.02.702952). 

```
Large, Annabel, and Ian Holmes. "Nested birth-death processes are competitive with parameter-heavy neural networks as time-dependent models of protein evolution." bioRxiv (2026): 2026-02.
```

A link to the PMLR version is forthcoming.

---
## Requirements
All implementation are written in Jax. Pytorch provides dataloaders and a minimal tensorboard interface. Key packages are: Python 3.9, JAX 0.4.28, Flax 0.8.3, PyTorch 2.2.2. A full conda environment is provided in `conda_env.yaml`.

```bash
conda env create -f conda_env.yaml
conda activate env
```

Training and evaluation require a single GPU. Run commands in `pair_alignment/` with prefix:

```bash
CUDA_VISIBLE_DEVICES=0, ...
```

---

## Directory Overview

| Directory | Contents |
|---|---|
| `preprocess_data/` | Full pipeline from raw Pfam v36.0 seed alignments and phylogenic trees to training-ready arrays |
| `conditional_frequencies_baseline/` | Code to run empirical conditional column frequency baseline |
| `figures_tables/` | Train, dev, and test set log-likelihoods for all experiments and all data partitions; script to reproduce Figure 1 from the paper |
| `replicate_training/` | Data partition definitions and training configs for all reported experiments |
| `pair_alignment/` | JAX/Flax training and evaluation code for all model families |

---

## Pfam cherries dataset

All data is constructed from Pfam v36.0 seed alignments and trees. We extract pairwise alignments and branch lengths from closest siblings in the phylogenetic trees (i.e. "cherries"). It is divided into 10 splits, plus an unused out-of-distribution (OOD) validation set comprising the widest and gappiest alignments. Three train/dev/test partitions are created:

**Partitions used in the paper:**

| Partition | Splits in train set | Splits in dev set | Splits in test set |
|---|---|---|---|
| 1 | 0–6 | 7 | 8, 9 |
| 2 | 1, 3–5, 7–9 | 6 | 0, 2 |
| 3 | 1, 2, 4, 6–9 | 0 | 3, 5 |

Per-split Pfam and clan membership is listed in `replicate_training/pfams_clans_per_split/`.

**Alphabet:** 20 amino acids (tokens 3–22) plus special tokens `<pad>=0`, `<bos>=1`, `<eos>=2`; gap `'.'=43`.

---

## Preprocessing

See `preprocess_data/README.txt` for the full pipeline (Pfam seed alignments → FastTree trees → CHERRIES arrays). Key parameters used in the paper:

| Parameter | Value |
|---|---|
| `num_splits` | 10 |
| `rand_key` | 6 |
| `topk1_valid` | 3 |
| `topk2_valid` | 8 |
| `alphabet_size` | 20 |

Raw data: Pfam v36.0 seed alignments, available via FTP at `ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam36.0/`.

---

## Training, Evaluation

See `pair_alignment/README.txt` for full documentation. All commands run from the project root; paths in configs are resolved relative to it.

```bash
# Unzip example data and configs to the project root
unzip pair_alignment/example_data.zip -d .
unzip pair_alignment/example_configs.zip -d .

# Train from scratch
CUDA_VISIBLE_DEVICES=0 python pair_alignment -task train -configs path/to/config.json

# Evaluate with saved parameters
CUDA_VISIBLE_DEVICES=0 python pair_alignment -task eval -configs path/to/eval_config.json
```
Configs for every model and partition reported in the paper are in `replicate_training/configs_used.zip`. Final parameters provided upon request, as these files exceed github's storage limit.

---

## Reported Results

### Log-likelihoods

Log-likelihoods for all models and all data partitions can be found in `figures_tables/`:

```
conditional_frequencies_baseline_{train,test}_set_loglikes.tsv
simple_indel_models_{train,test}_set_loglikes.tsv
hierarchical_mixture_models_{train,test}_set_loglikes.tsv
neural_models_{train,dev,test}_set_loglikes.tsv
```

### Figure 1 (AIC vs. number of parameters)

Figure 1 in the paper was generated from the hierarchical mixture log-likelihoods. Recreate it from provided TSVs with

```bash
cd figures_tables
python plot_figure_1.py
```

### Conditional frequency baseline

`conditional_frequencies_baseline_{train,test}_set_loglikes.tsv` comes from a parameter-free baseline that estimates a 21×21 conditional frequency matrix based on observed amino acid counts in the training data. Code to replicate this baseline can be found in `conditional_frequencies_baseline`.

```bash
cd conditional_frequencies_baseline
python run_frequency_baseline.py --example   # runs on bundled example data
python run_frequency_baseline.py             # runs on full Pfam cherries dataset
```

### Progen2 comparison
The Progen2 repo can be found at: [https://github.com/enijkamp/progen2](https://github.com/enijkamp/progen2). Reported scores specifically come from the "left-to-right" autoregressive likelihoods found in [likelihood.py](https://github.com/enijkamp/progen2/blob/9b4d4fb5ec19c9e55c4bb06305c0e613e46c1cf5/likelihood.py#L289).

```python
ll_lr_sum = ll(tokens=args.context, reduction='sum')
```
