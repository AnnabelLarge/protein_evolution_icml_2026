Preprocess Data:
================
This pipeline preprocesses Pfam-A.seed (multiple sequence alignments and
phylogenetic trees from Pfam v36.0) into training inputs for a pairHMM-based
model. The pipeline runs in five stages:

  1. Initial cleaning   -- filters MSAs: removes short/invalid peptides,
                           duplicate sequences (within and across families),
                           and prunes phylogenetic trees to match cleaned MSAs.
  2. Split & cherry-pick -- partitions Pfam families into train splits and an
                            out-of-distribution (OOD) validation set (selected
                            by alignment width and gap fraction), then extracts
                            "cherries" (leaf pairs) from each tree.
  3. Feature generation -- encodes aligned and unaligned sequences as numpy
                           arrays, paired with tree and alignment metadata.
  4. Count precomputation -- uses JAX to batch-compute substitution, insertion,
                             deletion, and transition counts needed for pairHMM
                             inputs (this step can be slow on large datasets).
  5. Concatenation & teardown -- concatenates per-part arrays into per-split
                                 .npy files and .tsv metadata, moves all
                                 outputs into DATA/ and intermediates.tar.gz.


Outputs:
--------

After the pipeline completes, two top-level directories are created:

  DATA/
    *_aligned_mats.npy        -- alignments
    *_seqs_unaligned.npy      -- unaligned sequences
    *_AAcounts.npy            -- amino acid emission counts
    *_AAcounts_subsOnly.npy   -- emission counts from substitution columns only
    *_subCounts.npy           -- substitution counts
    *_insCounts.npy           -- insertion counts
    *_delCounts.npy           -- deletion counts
    *_transCounts.npy         -- M/I/D/S/E transition counts
    *_metadata.tsv            -- per-sample metadata
    *_longest_alignment.txt   -- length of the longest alignment in the split
    *_longest_seqs.txt        -- length of the longest unaligned seq in split
    DATA/info/pfams_in_*      -- list of Pfam families in each split

  intermediates.tar.gz        -- intermediate files (cherry folders, removal
                                 logs, duplicate reports) compressed for audit


Requirements:
-------------

  Databases / external tools:
  - Pfam v36.0 seed file: ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam36.0/
  - FastTree 2.1.11 (No SSE3 build): used to impute missing phylogenetic trees

  Python packages:
  - Python 3.9.18  
  - JAX 0.4.28     -- batch computation of pairHMM transition/emission counts
  - Biopython 1.81 -- parsing and pruning phylogenetic trees


Arguement for clean_data.py:
----------------------------

  Required:
    -pfam_seed_file   Path to the Pfam-A.seed file (or example seed file)
    -tree_dir         Directory containing per-family .tree files

  Optional:
    -num_splits       Number of training splits (default: 10)
    -topk1_valid      Number of widest Pfam families held out for OOD valid
                      (default: 3; set to 0 to skip)
    -topk2_valid      Number of gappiest Pfam families held out for OOD valid
                      (default: 8; set to 0 to skip)
    -rand_key         Random seed for split assignment (default: 6)
    -metadata_header  Header string added to output stats file (default: metadata)
    -alphabet_size    Amino acid alphabet size (default: 20)
    -max_len          Maximum sequence length for padding (default: 5000)
    -batch_size       Batch size for count precomputation (default: 1000)


Quickstart:
-----------

Unzip EXAMPLE_INPUTS.zip, then run:

    python clean_data.py \
        -pfam_seed_file EXAMPLE_INPUTS/EXAMPLE_Pfam-A.seed \
        -tree_dir EXAMPLE_INPUTS/trees/ \
        -num_splits 2 \
        -topk1_valid 0 \
        -topk2_valid 0

The example uses -topk1_valid 0 and -topk2_valid 0 because the example dataset
is too small to hold out families for OOD validation.
