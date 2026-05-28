mkdir intermediates
mv CHERRIES_valid_all_metadata intermediates/
mv CHERRIES_valid_full_length intermediates/
mv CHERRIES_valid_summarized_counts intermediates/

mv CHERRIES_split*_all_metadata intermediates/
mv CHERRIES_split*_full_length intermediates/
mv CHERRIES_split*_summarized_counts intermediates/

mv CHERRIES-FROM_*/ intermediates/
mv REMOVED-PFAMS.tsv  intermediates/
mv repeats_ACROSS_families.tsv intermediates/
mv repeats_WITHIN_families.tsv intermediates/
mv SHORT-PEPS_INVALID-CHARS.tsv intermediates/
tar -czf intermediates.tar.gz intermediates/

mkdir DATA
mkdir DATA/info
mv pfams_in_* DATA/info

mv *AAcounts_subsOnly.npy DATA
mv *AAcounts.npy DATA
mv *aligned_mats.npy DATA
mv *delCounts.npy DATA
mv *insCounts.npy DATA
mv *seqs_unaligned.npy DATA
mv *subCounts.npy DATA
mv *transCounts.npy DATA
mv *metadata.tsv DATA
mv *longest_alignment.txt DATA
mv *longest_seqs.txt DATA
