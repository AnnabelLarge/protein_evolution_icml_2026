#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Bio import Phylo
import copy
import pandas as pd
from itertools import combinations
import os
from tqdm import tqdm

from utils.utils import make_sub_folder


class CherryPickr:
    def __init__(self, tree):
        self.original_tree = tree
        self.cherrytree = copy.deepcopy(tree)
        self.all_pairs = []
    
    
    def get_lowest_cherries(self):
        """
        generate the list of cherries remaining
        a cherry looks like-
        
          |---fruit 1
        --|               
          |---fruit 2
          
        """
        lowest_cherries = []
        for clade_obj in self.cherrytree.get_nonterminals():
            if len(clade_obj.get_terminals()) == 2:
                lowest_cherries.append(clade_obj)
        return lowest_cherries
    
    
    def find_distance_to_root(self, node):
        """
        calculate the full distance from this node to the root
        
        have to calculate based on the tree actively getting pruned, but
          this shouldn't matter, because I'm pruning from the bottom up...
        """
        path = self.cherrytree.get_path(target=node)
        total_dist = 0
        for clade_obj in path:
            total_dist += clade_obj.branch_length
        return total_dist
        
        
    def process_pair_tuple(self, stem_length, tup):
        """
        given a pair of clade objects and their original stem length,
        write results to a dictionary
        
        stem length is TOTAL DISTANCE from inner node to ROOT
        
        fruit1 length is the distance from inner node to fruit1
        fruit2 length is the distance from inner node to fruit2
        
         |--------(whatever)
         |
        -|        |--------(whatever)
         |        |
         |--------|            |---fruit1
                  |------------|
                               |--------fruit2
                                
        <---- stem length ---->
                              <---> fruit1 length
                              <--------> fruit 2 length
                              
        """
        assert len(tup) == 2, print(tup)
        
        ### store pairs in the following manner:
        # col1: stem length
        # col2: fruit 1
        # col3: length of fruit 1
        # col4: fruit 2
        # col5: length of fruit 2
        # col6: distance between fruit1 and fruit 2
        fruit_1 = tup[0].name
        fruit_1_len = tup[0].branch_length
        fruit_2 = tup[1].name
        fruit_2_len = tup[1].branch_length
        
        tree_dists = {'seq1': fruit_1,
                      'seq2': fruit_2,
                      'TREEDIST_stem-to-root': stem_length,
                      'TREEDIST_seq1-to-root': stem_length + fruit_1_len,
                      'TREEDIST_seq2-to-root': stem_length + fruit_2_len,
                      'TREEDIST_seq1-to-seq2': fruit_1_len + fruit_2_len}
        self.all_pairs.append(tree_dists)
        
    
    def process_pair_clade(self, inner_node, prune):
        """
        given one clade object that has two children, unpack it, get
        the stem length, and run self.process_pair_tuple
        """
        ### unpack the clade object
        stem_length = self.find_distance_to_root(inner_node)
        cherries = inner_node.get_terminals()
        
        ### process with previously defined method
        self.process_pair_tuple(stem_length, cherries)
        
        ### remove from tree if possible
        # this needs skipped if pair is last bit of the tree
        if prune:
            self.cherrytree.prune(cherries[0])
            self.cherrytree.prune(cherries[1])
    
    
    def break_triplet(self, inner_node):
        """
        If you have a triple (i.e. one inner node with three children),
        keep the pair that's got the smallest distance in the tree, and 
        toss the remainder
        """
        ### for three leaves coming to one inner node, make a pair out of
        ### the closest in the tree, then leave out the odd one
        triple = inner_node.get_terminals()
        
        ### find distance between all possible pairs
        best_pair = ()
        smallest_dist = 1e9
        for possible_pair_idxes in combinations(triple,2):
            len1 = possible_pair_idxes[0].branch_length
            len2 = possible_pair_idxes[1].branch_length
            dist = abs(len1 + len2)
            
            if dist < smallest_dist:
                best_pair = possible_pair_idxes
                smallest_dist = dist
        
        ### process the best pair (which is NOT in a clade object)
        # you're already at the root, so stem length is zero
        self.process_pair_tuple(stem_length = 0, tup = best_pair)
        
        
    def main_fn(self):
        """
        iterate through a tree and harvest pairs of related sequences from
        bottom to top of tree
        
        If tree ends in triple, will find the closest pair from a triplet, and 
          toss the remainder
        
        If tree ends in a single, will toss it
        """
        ################################################################
        ### PART 1: keep extracting all possible pairs, until there's  #
        ### three leaves left                                          #
        ################################################################
        while self.cherrytree.count_terminals() > 3:
            
            # get the current list of cherries in the tree
            cherrylst = self.get_lowest_cherries()
            
            # process the last one
            self.process_pair_clade(cherrylst[-1], prune=True)
                
            
        
        ################################################################
        ### PART 2: lots of weird tree structures and combinations of  #
        ### Tree and Clade objects can occur at the end of parsing     #
        ################################################################
        ### 2.1: if there's only one leaf left, don't need it; this should 
        ### also catch the case where len(self.cherrytree.get_nonterminals())
        ### is zero
        if self.cherrytree.count_terminals() == 1:
            print('Tossing last sequence')
        
        
        ### 2.2: IF there are multiple nonterminals still left in the tree, 
        ### then there's probably a pair and a single; process the pair 
        ### without pruning, and toss the remainder
        elif len(self.cherrytree.get_nonterminals()) == 2:
            last_cherry = self.get_lowest_cherries()[0]
            self.process_pair_clade(last_cherry, prune = False)
        
        ### 2.3: If there's one last nonterminal left, could have a couple
        ### things happening
        elif len(self.cherrytree.get_nonterminals()) == 1:
            final_nonterm = self.cherrytree.get_nonterminals()[0]
            
            ### 2.3.1: if there's two leaves, it's a normal pair; process
            ### as such
            if self.cherrytree.count_terminals() == 2:
                self.process_pair_clade(final_nonterm, prune=False)
            
            ### 2.3.2: if there's three leaves coming into one nonterminal
            ### node, then process with special function
            elif self.cherrytree.count_terminals() == 3:
                self.break_triplet(final_nonterm)
            
        ### keep this here, in case an edge case isn't caught...
        else:
            print(len(self.cherrytree.get_nonterminals()))
            print(self.cherrytree.count_terminals())
            assert False, 'Edge case not handled!'
        
        
    def output_pairs(self):
        """
        output a dataframe of pairs
        """
        out_df = pd.DataFrame(self.all_pairs)
        return out_df

     
def greedy_cherry_picker(tree_file: str,
                         tree_dir: str,
                         draw_trees: bool=False,
                         setup_folders: bool=False):
    """
    pick cherries for all trees in the tree_dir
    
    when I did my data processing, I split trees into subfolders to run in 
      parallel on HPC cluster (using this function)
     
     
    inputs:
    -------
        - tree_file (str): name of the tree file (PF#####.tree)
        - tree_dir (str): where individual .tree files are found
        - draw_trees (bool): optionally, draw the tree to a text file
        - setup_folders (bool): optionally, try making the output folders
     
    returns:
    --------
        (None)
     
    outputs:
    --------
        - folder of metadata about which pairs are cherries, with
          tree distances from FastTree initial estimates
        - (optionally) folder of tree drawings in text files
    """
    ### setup folders, if desired
    cherry_metadata_folder = f'CHERRIES-FROM_{tree_dir}'
    cherry_metadata_folder = cherry_metadata_folder.replace('/trees','')
    if setup_folders:
        make_sub_folder(in_dir = '.', 
                        sub_folder = cherry_metadata_folder)
    
    ### run cherry picker
    # load tree into bio object
    tree = Phylo.read(f"./{tree_dir}/{tree_file}", "newick")

    # use the picker object
    pickr = CherryPickr(tree)
    pickr.main_fn()
    out_df = pickr.output_pairs()
    
    # save the dataframe
    pfam = tree_file.replace('.tree','')
    out_df.to_csv(f'{cherry_metadata_folder}/{pfam}_cherries.tsv', 
                  sep='\t')
    
    
    ### optionally, draw the tree
    if draw_trees:
        drawings_folder = f'ASCII_{tree_dir}'

        if setup_folders:
            make_sub_folder(in_dir = '.', 
                            sub_folder = drawings_folder)
        
        with open(f'{drawings_folder}/{pfam}_ascii-tree.txt','w') as g:
            Phylo.draw_ascii(tree, file=g)
        
        
def pick_trees_serially(tree_dir: str,
                        draw_trees: bool=False): 
    """
    use greedy_cherry_picker() on all files in a directory
    """
    file_lst = [f for f in os.listdir(tree_dir) 
                if f.startswith('PF') and f.endswith('.tree')]
    
    # with the first file, may need to setup directories
    greedy_cherry_picker(tree_file = file_lst[0],
                         draw_trees = draw_trees,
                         tree_dir = tree_dir,
                         setup_folders = True)
    
    # with subsequent files, output directories are already ready
    for tree_file in tqdm(file_lst[1:]):
        greedy_cherry_picker(tree_file = tree_file,
                             draw_trees = draw_trees,
                             tree_dir = tree_dir,
                             setup_folders = False)
