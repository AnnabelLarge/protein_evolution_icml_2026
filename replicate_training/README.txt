Replicate training:
====================

Follow these steps to replicate all models from the paper.


1.) acquire data, clean:
-------------------------
FTP from Pfam v36.0: ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam36.0/

postprocess using scripts in ../preprocess_data; we use the following values-
  > num_splits: 10
  > rand_key: 6
  > topk1_valid: 3
  > topk2_valid: 8
  > alphabet_size: 20


2.) partition data into train-dev-test:
-----------------------------------------
Split IDs: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (10 total)

See pfams_clans_per_split/ for names of specific pfams and clans in each split

Partition 1:
	train: 0, 1, 2, 3, 4, 5, 6
	dev: 7
	test: 8, 9

Partition 2:
	train: 1, 3, 4, 5, 7, 8, 9
	dev: 6
	test: 0, 2

Partition 3:
	train: 1, 2, 4, 6, 7, 8, 9
	dev: 0
	test: 3, 5


3.) train models
------------------
Configs for all data partitions and all models are in ./configs_used.zip.
Unzip and use with the training code in ../train_models.

Config subdirectories:

  basic_neural_models/
      Feedforward neural sequence-to-sequence models (no indel head).
      Architectures: CNN-s2s, LSTM-s2s, Transformer-s2s (1-block and 6-block).

  neural_tkf_models/
      Neural sequence embedders with a TKF indel prediction head.
      Architectures: CNN-TKF, LSTM-TKF, Transformer-TKF (1-block and 6-block).

  mixture_of_site_classes/
      Classical pairHMM (F81 substitution + TKF91 or TKF92 indels) with
      mixtures of rate classes across sites (1, 2, 3, 4, 5, 10, 20, 30, 175,
      500, 900 classes).

  mixture_of_fragment_classes/
      TKF92 pairHMM with mixtures of fragment + site classes (2, 3, 4, 5,
      10, 20, 30 fragment classes).

  mixture_of_domain_classes/
      PairHMM with domain-level class mixtures (2, 3, 4, 5, 10 domain
      classes), F81 substitution + TKF92 indels.

  GGI_indel_models/
      Classical pairHMMs (TKF91, TKF92, LG05, RS07, H20 indel models) with
      either F81 or GTR-LG08 substitution. Three branch-length modes and 
      two substitution models:
        f81_use_pfam_branch_lengths/      -- one branch length per sample
        gtr-lg08_use_pfam_branch_lengths/  -- one branch length per sample
        gtr-lg08_marginalize_over_grid_of_times/  -- marginalise over a time grid
