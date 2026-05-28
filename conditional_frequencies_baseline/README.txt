Conditional Frequencies Baseline
=================================
A simple nonparametric method that scores pairwise alignments using
empirical pairwise alignment column frequencies estimated from training data.

Given a training set of pairwise alignments, the protocol is:

1. Build a 21 x 21 joint frequency matrix from counts 
   C(ancestor character, descendant character). Note that this
   method treats gap as a member of the sequence alphabet and
   foregoes evolutionary time altogether.

2. Score pairwise alignments with 1.) the joint frequency matrix
   to calculate P(ancestor, descendant), and 2.) a term to indicate
   how likely the alignment length is, assuming a geometric distribution.

3. Build a 21 element frequency vector from marginal counts 
   C(ancestor character).

4. Score ancestral sequences with 1.) the marginal frequency vector,
   anc 2.) a term to indicate how likely the ancestral sequence 
   length is, assuming a separate geometric distribution.

5. Calculate conditional probabilites by dividing the joint probability
   by the marginal probability
   P(descendant | ancestor) = P(ancestor, descendant) / P(ancestor)


Quick start (example data)
--------------------------
    python run_frequency_baseline.py --example

This runs on the two splits in example_data/ (split 0 = train, split 1 = test)
and writes output files prefixed "example_" to the current directory.


Full run (Pfam cherries dataset)
------------------------------------
    python run_frequency_baseline.py

Expects the following layout in the current directory:

    DATA_cherries/
      FAMCLAN-CHERRIES_split{0..9}_aligned_mats.npy
      FAMCLAN-CHERRIES_split{0..9}_metadata.tsv
      FAMCLAN-CHERRIES_OOD_Valid_aligned_mats.npy
      FAMCLAN-CHERRIES_OOD_Valid_metadata.tsv

Emission counts are written to the existing directory pairhmm_counts/
(create this first). 


Input file format
-----------------
*_aligned_mats.npy
    (N, L, C=4) int array of aligned sequence pairs; channels 0 and 1 are
    ancestor and descendant token indices.

*_metadata.tsv
    TSV with columns including pairID, ancestor, descendant, anc_seq_len,
    alignment_len, pfam.


Outputs
-------
For each analysis run (prefix <p>), outputs will be:

  <p>_geom_p_eos.tsv
      Estimated geometric end-of-sequence parameters.

  <p>_{train,test}-set_loglikes-per-samp.tsv
      Per-pair log-likelihoods (joint, ancestral marginal, conditional).

  <p>_{TRAIN,TEST}-AVE-LOGLIKES.tsv
      Summary statistics (average loss, ECE, perplexity).
