#!/usr/bin/env python3
# coding: utf-8

# CLUSTERING TOOLS
# Development of an MHC-I-peptide binding predictor using Monte Carlo simulations (Vallejo-Vallés et al. 2023)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bioprospecting
import os
import shutil
import mdtraj as md

from Bio.PDB.Polypeptide import one_to_three, three_to_one
from Bio.Align import substitution_matrices
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import whiten


### SEQUENCE CLUSTERING ####

def blosumMatrix(sequences):

    """
    Returns a BLOSUM matrix (similarity matrix).

    PARAMETERS
    ------------------------------------------------------------
    · sequences : list of sequences of equal length (for example 9-mers)
    ------------------------------------------------------------
    """

    # Load BLOSUM62 matrix
    blosum62 = substitution_matrices.load("BLOSUM62")

    # Generate dictionary
    aa_position = {x[1]:x[0] for x in enumerate(blosum62.alphabet)}

    # Generate empty matrix to store blosum valuess
    bm = np.zeros((len(sequences), len(sequences)))

    for i in range(len(sequences)): # iterate in one axis

        for j in range(len(sequences)): # iterate the other axis

            if i >= j: # treballar amb un quadrant

                assert len(sequences[i]) == len(sequences[j]) # check if equal axis length

                score = 0
                for p in range(len(sequences[i])): # iterate each aminoacid of the sequence
                    AApi = sequences[i][p]
                    AApj = sequences[j][p]
                    pi = aa_position[AApi]
                    pj = aa_position[AApj]
                    score += blosum62[pi][pj]
                bm[i][j] = score # fill one quadrant
                bm[j][i] = bm[i][j] # fill the other quadrant with the same information

    return bm



def distanceMatrix(similarity_matrix):

    """
    Returns a distance matrix (disimilarity matrix).

    PARAMETERS
    ------------------------------------------------------------
    · similarity_matrix : BLOSUM matrix from the function blosumMatrix
    ------------------------------------------------------------
    """

    disimilarity_matrix = similarity_matrix*-1

    disimilarity_matrix = disimilarity_matrix + np.abs(disimilarity_matrix.min())

    # Check that numbers in the diagonal are the lower ones
    # If true it won't give an assertion error
    for i in range(disimilarity_matrix.shape[0]):
        for j in range(disimilarity_matrix.shape[1]):
            assert disimilarity_matrix[i][i] <= disimilarity_matrix[i][j]

    return disimilarity_matrix


def clusteringPlots(bm,dm):

    """
    Prints plots to analyse clustering quality.

    PARAMETERS
    ------------------------------------------------------------
    · bm : BLOSUM matrix from the function blosumMatrix
    · dm : distance matrix
    ------------------------------------------------------------
    """

    # Plot matrices
    plt.matshow(bm)
    plt.title('BLOSUM Matrix')
    plt.show()

    plt.matshow(dm)
    plt.title('Distance Matrix')
    plt.show()

    # Plot correlation between matrices
    plt.scatter(bm.flatten(),dm.flatten())
    plt.title('Correlation between matrices')
    plt.show()


### RMSD CLUSTERING ###

def clusteringRMSD(trajectories_aligned, structural_labels):

    peptide_trajectories = md.join(trajectories_aligned,check_topology=False)

    frames = len(peptide_trajectories)
    matrix_shape = (frames,frames)
    rmsd_matrix = np.zeros(matrix_shape)

    for i in range(frames):
        rmsd_matrix[i] = md.rmsd(peptide_trajectories, peptide_trajectories, frame=i)

    clustering_rmsd = bioprospecting.clustering.hierarchical.cluster(rmsd_matrix)
    centroid = clustering_rmsd.getNClusters(1,return_centroids=True)
    print('Labels: ',structural_labels)
    print('Centroid: ',centroid)
    print('Label centroid: ',structural_labels[centroid[1]])
    clustering_rmsd.plotDendrogram(figsize=(20, 5),dpi=100,labels=structural_labels)

    return structural_labels[centroid[1]]


def superposingStructures(pdbs):
    if os.path.exists('HLA-A_0201_seqvariabilitysuperposedrmsd')==False:
        os.makedirs('HLA-A_0201_seqvariabilitysuperposedrmsd')

    # Use the first pdb as reference for superposing structures
    path_ref = 'HLA-A_0201/'+pdbs[0]+'.pdb'
    reference = md.load_pdb(path_ref)
    print('Reference PDB: ', path_ref)

    trajectories_aligned = []
    labels = []
    pep_sequences = dict()

    for i in pdbs:
        path_others = 'HLA-A_0201/'+i+'.pdb'
        target = md.load_pdb(path_others)
        print('Target PDB: ', path_others)

        # Get 9-mers:
        sequence = ''
        target_peptide_residues = target.topology.chain(2).residues # peptide chain is always chain 2
        for residue in target_peptide_residues:
            sequence += three_to_one(residue.name)
        print(sequence)

        if len(sequence)==9:
            try:
                superposeTrajectories(target,reference,chains=[0,1,2])
                peptide_CA = target.topology.select('chainid 2 and name CA')  # only c-alpha and peptide chain
                trajectories_aligned.append(target.atom_slice(peptide_CA))
                path_sup = 'HLA-A_0201_seqvariabilitysuperposedrmsd/' + i+'.pdb'
                target.save_pdb(path_sup)
                print('Saving superposed pdb: ',path_sup)
                labels.append(i)
                pep_sequences[i] = sequence
            except ValueError:
                print(' ')

    return trajectories_aligned,labels


def superposeTrajectories(trajectory, reference, chains):

    # Get dictionaries linking sequence positions to residue chain_indexes
    trajectory_indexes = getChainIndexesToResidueIndexes(trajectory.topology)
    reference_indexes = getChainIndexesToResidueIndexes(reference.topology)

    # Align sequences and store common residues
    trajectory_residues = []
    reference_residues = []
    for i in range(len(chains)):
        sequences = {}
        # Get corresponding chain ids
        c = chains[i]

        # Store sequences into a dictionary
        sequences['target'] = getTopologySequence(trajectory.topology, c)
        sequences['reference'] = getTopologySequence(reference.topology, c)

        # Align sequences
        alignment = bioprospecting.alignment.mafft.multipleSequenceAlignment(sequences,quiet=True)

        # Get coincident positions in the alignment
        positions = getCommonPositions(alignment[0].seq, alignment[1].seq)

        # Store common residues
        for p in positions:
            trajectory_residues.append(trajectory_indexes[c][p[0]])
            reference_residues.append(reference_indexes[c][p[1]])

    # Store common alpha-carbon atoms
    trajectory_atoms = [ a.index for a in trajectory.topology.atoms if a.name == 'CA'\
                         and a.residue.index in trajectory_residues ]
    reference_atoms = [ a.index for a in reference.topology.atoms if a.name == 'CA' \
                        and a.residue.index in reference_residues ]

    # Align trajectory
    trajectory.superpose(reference, atom_indices=trajectory_atoms, ref_atom_indices=reference_atoms)


def getChainIndexesToResidueIndexes(topology):
    residue_indexes  = {}
    for chain in topology.chains:
        residue_indexes[chain.index] = {}
        for i,residue in enumerate(chain.residues):
            residue_indexes[chain.index][i] = residue.index
    return residue_indexes


def getTopologySequence(topology, chain_index, non_protein='X'):
    sequence = ''
    for r in topology.chain(chain_index).residues:
        try:
            sequence += three_to_one(r.name)
        except:
            sequence += non_protein
    return sequence


def getCommonPositions(sequence1, sequence2, mode='exact'):
    positions = []
    # Initialize idependent counters for sequence positions
    s1p = 0
    s2p = 0
    # Iterate thorugh the aligned positions
    for i in range(len(sequence1)):
        # Compare sequences according to selected mode
        if sequence1[i] != '-' and sequence2[i] != '-':
            if mode == 'exact':
                if sequence1[i] == sequence2[i]:
                    positions.append((s1p, s2p))
            elif mode == 'aligned':
                positions.append((s1p, s2p))
        # Add to position counters
        if sequence1[i] != '-':
            s1p += 1
        if sequence2[i] != '-':
            s2p += 1

    return positions


def getMinimizedPDB(substring):
    minimized_pdbs = os.listdir('HLA-A_0201_bestmodels/')
    for file in filter (lambda x: substring in x, minimized_pdbs):
        return file
