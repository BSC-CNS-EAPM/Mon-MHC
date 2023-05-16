#!/usr/bin/env python3
# coding: utf-8

# MAIN PROGRAM
# Development of an MHC-I-peptide binding predictor using Monte Carlo simulations (Vallejo-Vallés et al. 2023)

import argparse
import math
import os
import random
import re
import shutil
import time

import numpy as np

from pyrosetta import *
from pyrosetta.toolbox import cleanATOM
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, NeighborhoodResidueSelector
from pyrosetta.rosetta.core.scoring import CA_rmsd

pyrosetta.init('-mute all') # by default it repeats 5 times


def getArgs():

    """
    Get arguments from the user.
    """

    parser = argparse.ArgumentParser(description = "Arguments necessary for the main program")

    parser.add_argument('--b2m_chain',
                            dest = 'b2m_chain', action = 'store',
                            default = 'B',
                            help = 'Chain identifier of the B2M in the input complex.',
                            type = str)
    parser.add_argument('--centroid_mhc',
                            dest = 'centroid_mhc', action = 'store',
                            default = 14,
                            help = 'Residue closer to the centroid of the MHC chain.',
                            type = int)
    parser.add_argument('--displace_peptide',
                            dest = 'displace_peptide', action = 'store',
                            default = 0,
                            help = 'Displace peptide chain (A).',
                            type = int)
    parser.add_argument('--energy_threshold',
                            dest = 'energy_threshold', action = 'store',
                            default = 0.5,
                            help = 'Energy threshold to consider that energies have converged during minimization.',
                            type = float)
    parser.add_argument('--fastrelax_repeats',
                            dest = 'fastrelax_repeats', action = 'store',
                            default = 1,
                            help = '',
                            type = int)
    parser.add_argument('--input_complex',
                            dest = 'input_complex', action = 'store',
                            default = None,
                            help = 'An input pdb file containing the ternary complex.',
                            type = str)
    parser.add_argument('--mhc_chain',
                            dest = 'mhc_chain', action = 'store',
                            default = 'A',
                            help = 'Chain identifier of the MHC in the input complex.',
                            type = str)
    parser.add_argument('--minimization_steps',
                            dest = 'minimization_steps', action = 'store',
                            default = 5,
                            help = 'Number of steps for the Fast Relax minimization, previous to the Monte Carlo simulation.',
                            type = int)
    parser.add_argument('--montecarlo_steps',
                            dest = 'montecarlo_steps', action = 'store',
                            default = 100,
                            help = 'Number of steps for Monte Carlo simulation.',
                            type = int)
    parser.add_argument('--neighbours_distance',
                            dest = 'neighbours_distance', action = 'store',
                            default = 7.0,
                            help = 'Maximum distance (A) between residues to be considered as neighbours.',
                            type = float)
    parser.add_argument('--peptide_chain',
                            dest = 'peptide_chain', action = 'store',
                            default = 'C',
                            help = 'Chain identifier of the peptide in the input complex.',
                            type = str)
    parser.add_argument('--peptide_sequence',
                            dest = 'peptide_sequence', action = 'store',
                            default = None,
                            help = 'Peptide sequence that we want to have in the ternary complex.',
                            type = str)
    parser.add_argument('--pymol',
                            dest = 'pymol', action = 'store_true',
                            default = False,
                            help = 'Activate pymol.')
    parser.add_argument('--run_id',
                            dest = 'run_id', action = 'store',
                            default = '1',
                            help = 'Run number (useful when executing the script multiple times with bash script).',
                            type = str)
    parser.add_argument('--save_pdb',
                            dest = 'save_pdb', action = 'store_true',
                            default = False,
                            help = 'Save pdb files at each step.')
    parser.add_argument('--score_file',
                            dest = 'score_file', action = 'store',
                            default = 'score_file.sc',
                            help = 'Name of the output score file.',
                            type = str)
    parser.add_argument('--score_function',
                            dest = 'score_function', action = 'store',
                            default = 'ref2015',
                            help = 'Score function for the sampling algorithm.',
                            type = str)
    parser.add_argument('--shear_angle_helical',
                            dest = 'shear_angle_helical', action = 'store',
                            default = 10,
                            help = 'Maximum angle allowed for the backbone perturbations of the Shear mover (during Monte Carlo simulation).',
                            type = int)
    parser.add_argument('--shear_angle_sheet',
                            dest = 'shear_angle_sheet', action = 'store',
                            default = 10,
                            help = 'Maximum angle allowed for the backbone perturbations of the Shear mover (during Monte Carlo simulation).',
                            type = int)
    parser.add_argument('--shear_angle_loop',
                            dest = 'shear_angle_loop', action = 'store',
                            default = 10,
                            help = 'Maximum angle allowed for the backbone perturbations of the Shear mover (during Monte Carlo simulation).',
                            type = int)
    parser.add_argument('--small_angle_helical',
                            dest = 'small_angle_helical', action = 'store',
                            default = 10,
                            help = 'Maximum angle allowed for the backbone perturbations of the Small mover (during Monte Carlo simulation).',
                            type = int)
    parser.add_argument('--small_angle_sheet',
                            dest = 'small_angle_sheet', action = 'store',
                            default = 10,
                            help = 'Maximum angle allowed for the backbone perturbations of the Small mover (during Monte Carlo simulation).',
                            type = int)
    parser.add_argument('--small_angle_loop',
                            dest = 'small_angle_loop', action = 'store',
                            default = 10,
                            help = 'Maximum angle allowed for the backbone perturbations of the Small mover (during Monte Carlo simulation).',
                            type = int)
    parser.add_argument('--shear_moves',
                            dest = 'shear_moves', action = 'store',
                            default = 5000,
                            help = 'Number of moves for the backbone perturbations of the Shear mover (during Monte Carlo simulation) ',
                            type = int)
    parser.add_argument('--small_moves',
                            dest = 'small_moves', action = 'store',
                            default = 5000,
                            help = 'Number of moves for the backbone perturbations of the Small mover (during Monte Carlo simulation).',
                            type = int)
    parser.add_argument('--temperature',
                            dest = 'temperature', action = 'store',
                            default = 0.5,
                            help = 'Set up temperature (ºC) for the backbone movers (Small and Shear) and Fast Relax.',
                            type = float)
    return parser.parse_args()


dictionary_1to3 = {
 'A': 'ALA',
 'C': 'CYS',
 'D': 'ASP',
 'E': 'GLU',
 'F': 'PHE',
 'G': 'GLY',
 'H': 'HIS',
 'I': 'ILE',
 'K': 'LYS',
 'L': 'LEU',
 'M': 'MET',
 'N': 'ASN',
 'P': 'PRO',
 'Q': 'GLN',
 'R': 'ARG',
 'S': 'SER',
 'T': 'THR',
 'V': 'VAL',
 'W': 'TRP',
 'Y': 'TYR',
 'B': 'ASX',
 'X': 'XAA',
 'Z': 'GLX',
 'J': 'XLE',
 'U': 'SEC',
 'O': 'PYL'
}

def seq3_custom(seq, custom_map=None, undef_code="Xaa"):
    """
    Biopython's function seq3 customized to use without importing it (remove dependencies)
    """
    if custom_map is None:
        custom_map = {"*": "Ter"}
    threecode = dict(
        list(dictionary_1to3.items()) + list(custom_map.items())
    )
    return "".join(threecode.get(aa, undef_code) for aa in seq)


def seq1_custom(r3):
    """
    Converts amino acid naming from three letter code to one letter code.
    """

    r3 = r3.upper()
    r1 = [i for i in dictionary_1to3 if dictionary_1to3[i]==r3][0]

    return r1


def evaluateSequence(pose,chain,sequence=None):

    """
    This function checks that the peptide sequence given by the user
    has the same length than the peptide in the ternary complex,
    and that it is an aminoacid sequence.

    PARAMETERS
    ------------------------------------------------------------
    · pose: PyRosetta pose of the ternary complex.
    · sequence: peptide sequence given by the user (type: str).
    · chain: chain identifier of the peptide (type: str).
    ------------------------------------------------------------
    """

    # Count number of aminoacids in the peptide sequence (from ternary complex)
    count = 0
    for index,residue in enumerate(pose.residues):
        # Get chain identifier from pdb
        c = pose.pdb_info().pose2pdb(index+1).split()[-1]
        if c == chain: # peptide chain
            count += 1

    if sequence:
        # Length sequence
        try:
            assert count==len(sequence)
        except AssertionError:
            raise(AssertionError('Please provide a peptide sequence of %s residues.' % count))

        # Amino acid sequence
        try:
            aa = 'ACDEFGHIKLMNPQRSTVWY'
            assert all(i in aa for i in sequence)
        except AssertionError:
            raise(AssertionError('Please provide an aminoacid sequence.'))

    return count


def mutatePeptide(pose,sequence,chain):

    """
    This function mutates the peptide to the sequence of interest given by the user.
    It returns a list of the mutations and the PyRosetta pose with the peptide mutated.

    PARAMETERS
    ------------------------------------------------------------
    · pose: PyRosetta pose of the ternary complex.
    · sequence: peptide sequence given by the user (type: str).
    · chain: chain identifier of the peptide (type: str).
    ------------------------------------------------------------
    """

    count = 0
    mutations = []
    residue_index = 1

    MutateResidue = pyrosetta.rosetta.protocols.simple_moves.MutateResidue

    # Iterate in pose residues
    for index,residue in enumerate(pose.residues):
        # Get chain identifier from pdb
        c = pose.pdb_info().pose2pdb(index+1).split()[-1]

        if c != chain: # non-peptide chain
            reset_index = index

        elif c == chain: # peptide chain
            # Reset index, start again from 0:
            chain_index = index-reset_index-1
            # Transform residue name from 1 letter to 3 letter code (e.g. A - ALA)
            new_residue = seq3_custom(sequence[chain_index])

            if new_residue != residue.name()[:3]: # actual peptide residue doesn't match residue given by user
                # Change residue, apply to pose
                MutateResidue(target=index+1, new_res=new_residue[:3]).apply(pose)
                # Add mutation to list (e.g. M278A)
                mutation = seq1_custom(residue.name()[:3])   + str(residue_index) + sequence[chain_index]
                mutations.append(mutation)

            count += 1

            residue_index += 1

    return mutations, pose


def peptideRMSD(pose,pose_reference,chain):
    """
    This function returns the peptide RMSD, comparing the input pose vs the
    pose at each Monte Carlo step.

    PARAMETERS
    ------------------------------------------------------------
    · pose: PyRosetta pose of the ternary complex, at a certain Monte Carlo step.
    · pose_reference: PyRosetta pose of the input ternary complex with mutated peptide.
    · chain: chain identifier of the peptide (type: str).
    ------------------------------------------------------------
    """

    count = 0
    mutations = []
    residue_index = 1

    # Iterate in pose residues
    for index,residue in enumerate(pose.residues):
        # Get chain identifier from pdb
        c = pose.pdb_info().pose2pdb(index+1).split()[-1]

        if c == chain: # peptide chain
            # Reset index, start again from 0:
            first_position = index
            break


    chain_index = (first_position,first_position+length_peptide)

    rmsd = CA_rmsd(pose, pose_reference, start=chain_index[0], end=chain_index[1])

    return rmsd


# def my_RMSD(pose_reference,pose):
#     rmsd = rosetta.protocols.stepwise.modeler.align.superimpose_with_stepwise_aligner(pose_reference, pose)
#     return rmsd

def getCoordinates(pose, residues=None, bb_only=False, sc_only=False):
    """
    Get all the pose atoms coordinates. An optional list of residues can be given
    to limit coordinates to only include the atoms of these residues.

    Parameters
    ==========
    pose : pyrosetta.rosetta.core.pose.Pose
        Pose from which to get the atomic coordinates.
    residues : list
        An optional list of residues to only get their coordinates.
    bb_only : bool
        Get only backbone atom coordinates from the pose.
    sc_only : bool
        Get only sidechain atom coordinates from the pose.

    Returns
    =======
    coordinates : numpy.ndarray
        The pose's coordinates.
    """

    if bb_only and sc_only:
        raise ValueError('bb_only and sc_only cannot be given simultaneously!')

    coordinates = []
    for r in range(1, pose.total_residue()+1):
        # Check if a list of residue indexes is given.
        if residues != None:
            if r not in residues:
                continue

        # Get residue coordinates
        residue = pose.residue(r)
        bb_indexes = residue.all_bb_atoms()
        for a in range(1, residue.natoms()+1):

            # Skip non backbone atoms
            if bb_only:
                if a not in bb_indexes:
                    continue

            # Skip backbone atoms
            if sc_only:
                if a in bb_indexes:
                    continue

            # Get coordinates
            xyz = residue.xyz(a)
            xyz = np.array([xyz[0], xyz[1], xyz[2]])
            coordinates.append(xyz)

    coordinates = np.array(coordinates)

    return coordinates

def getCoordinates(pose, residues=None, bb_only=False, sc_only=False):
    """
    Get all the pose atoms coordinates. An optional list of residues can be given
    to limit coordinates to only include the atoms of these residues.

    Parameters
    ==========
    pose : pyrosetta.rosetta.core.pose.Pose
        Pose from which to get the atomic coordinates.
    residues : list
        An optional list of residues to only get their coordinates.
    bb_only : bool
        Get only backbone atom coordinates from the pose.
    sc_only : bool
        Get only sidechain atom coordinates from the pose.

    Returns
    =======
    coordinates : numpy.ndarray
        The pose's coordinates.
    """

    if bb_only and sc_only:
        raise ValueError('bb_only and sc_only cannot be given simultaneously!')

    coordinates = []
    for r in range(1, pose.total_residue()+1):
        # Check if a list of residue indexes is given.
        if residues != None:
            if r not in residues:
                continue

        # Get residue coordinates
        residue = pose.residue(r)
        bb_indexes = residue.all_bb_atoms()
        for a in range(1, residue.natoms()+1):

            # Skip non backbone atoms
            if bb_only:
                if a not in bb_indexes:
                    continue

            # Skip backbone atoms
            if sc_only:
                if a in bb_indexes:
                    continue

            # Get coordinates
            xyz = residue.xyz(a)
            xyz = np.array([xyz[0], xyz[1], xyz[2]])
            coordinates.append(xyz)

    coordinates = np.array(coordinates)

    return coordinates


def displacePeptide(pose,chain,distance):
    """
    Move the native peptide away from the binding groove.

    PARAMETERS
    ------------------------------------------------------------
    · pose: pose object
    · chain: chain of the pose object
    ------------------------------------------------------------
    """

    # Remove all Atom Pair constraints from a pose
    rosetta.core.scoring.constraints.remove_constraints_of_type(pose, 'AtomPair')
    # Get peptide jump id
    jump_id = rosetta.core.pose.get_jump_id_from_chain(chain,pose)
    # Get peptide chain id
    chain_id = rosetta.core.pose.get_chain_id_from_chain(chain,pose)
    # Get peptide residues
    peptide_residues = rosetta.core.pose.get_chain_residues(pose,chain_id)
    # Get residues ids (to use in getCoordinates)
    peptide_residues_id = [x.seqpos() for x in list(peptide_residues)]
    protein_residues_id = [r for r in range(1,pose.total_residue()) if r not in  peptide_residues_id]
    # Get peptides coordinates
    peptides_coor = getCoordinates(pose,peptide_residues_id)
    # Get protein coordinates
    protein_coor = getCoordinates(pose,protein_residues_id)
    # Get centroids
    peptide_centroid = np.average(peptides_coor, axis=0)
    protein_centroid = np.average(protein_coor, axis=0)

    vector =  peptide_centroid - protein_centroid
    vector = vector/np.linalg.norm(vector)
    vector = rosetta.numeric.xyzVector_double_t(vector[0], vector[1], vector[2])

    # Displace peptide 20 A
    peptide_mover = rosetta.protocols.rigid.RigidBodyTransMover()
    peptide_mover.trans_axis(vector)
    peptide_mover.step_size(distance)
    peptide_mover.rb_jump(jump_id)
    peptide_mover.apply(pose)

    return pose


def calculateBindingScore(pose, chain):
    """
    Calculate binding score by displacing the peptide 1000 A from the binding pocket and calculating the energy.

    PARAMETERS
    ------------------------------------------------------------
    · pose : PyRosetta pose of the ternary complex, at a certain Monte Carlo step.
    · chain : chain identifier of the peptide (type: str).
    ------------------------------------------------------------
    """

    # Generate a clone (as we don't want to modify the original pose)
    clone = Pose()
    clone.assign(pose)

    # Get initial energy
    Ei = sfxn(clone)

    # Move peptide away from the Binding Groove
    clone = displacePeptide(clone,chain,1000)

    # Get final energy
    Ef = sfxn(clone)

    # Calculate Binding energy
    binding_energy = Ei-Ef

    return binding_energy


class dataReporter:
    """
    Writes down to a file the simulation data. This reporter class is tailor made
    for reporting data about peptide binding and catalytic triad.
    """
    def __init__(self, output_file, sfxn, peptide_chain, time=False):
        """
        Create a reporter file.

        Parameters
        output_file : str
            Name of the output reporter file.
        sfxn : str
            Rosetta scorefuntion used to evaluate the pose when reporting energetics.
        time : bool
            Should the reporter write down a time column for each report.
        """

        # Initialize variables
        self.output_file = output_file
        self.data = None
        self.time = time
        self.sfxn = sfxn
        self.peptide_chain = peptide_chain

        # Check if output file already exists
        if os.path.exists(self.output_file):
            self.data_exists = True
        else:
            self.data_exists = False

        # Define header format
        self.header = '{0:>11}{1:>13}{2:>16}'

        # Define mandatory labels
        self.header_labels = ['mc_step', 'total_score', 'binding_score']
        self.index = len(self.header_labels)-1 # Define current index position

        # Add headers for score terms
        for score in self.sfxn.get_nonzero_weighted_scoretypes():
            self.index += 1
            length = max(15, len(score.name))
            self.header += '{'+str(self.index)+':>'+str(length+1)+'}'
            self.header_labels.append(score.name)

        # Add time format and label
        if self.time:
            self.index += 1
            self.header += '{'+str(self.index)+':>12}'
            self.header_labels.append('time')

        # Add description labels
        self.index += 1
        self.header += '{'+str(self.index)+':>'+str(len(self.output_file)+5)+'}' # FIX
        self.header_labels.append('description')

        # Add peptide total RMSD label
        self.index += 1
        self.header += '{'+str(self.index)+':>'+str(len(self.output_file)+5)+'}' # FIX
        self.header_labels.append('total_rmsd')

        # Add peptide RMSD label
        self.index += 1
        self.header += '{'+str(self.index)+':>'+str(len(self.output_file)+5)+'}' # FIX
        self.header_labels.append('peptide_rmsd')
        self.header += '\n'


    def check_previous_run(self, last_struct, overwrite):
        """
        Check for previous data if an existing data report file data_exists.

        Parameters
        ==========
        last_struct : int
            The mc step of the last structure writte in the report silent file.
        overwrite : bool
            Overwrite any prviously written data.
        """

        # Check unfinished structures from previous runs
        if os.path.exists(self.output_file) and os.stat(self.output_file).st_size != 0 and overwrite == False:
            self.open('r')
            with self.data as f:
                lines = f.readlines()
                if (lines[0].strip().split()) != self.header_labels:
                    raise ValueError('Input parameters given do not match those of the existing file. Add overwrite argument or modify parameters')
                dsc_index = lines[0].strip().split().index('mc_step')

            self.open('w')
            for i,line in enumerate(lines):
                l = line.strip().split()
                if i == 0:
                    self.data.write(line)
                else:
                    if int(l[dsc_index]) <= last_struct:
                        self.data.write(line)
            self.data.flush()
        else:
            self.open('w')
            self.write_header()

    def add_extra_headers(self, headers):
        """
        Add extra headers for writting additional data columns to the file.

        Parameters
        ==========
        headers : list
            The list of strings representing the headers for the additional column
            data to write.
        """

        self.header = self.header.replace('\n', '')
        for header in headers:
            self.index += 1
            length = len(header)
            self.header += '{'+str(self.index)+':>'+str(length+1)+'}'
            self.header_labels.append(header)
        self.header += '\n'

    def write_header(self):
        """
        Write the headers to the report file (call only once when a new file is
        written all headers have been appended with add_extra_headers() if any.)
        """
        self.data.write(self.header.format(*self.header_labels))
        self.data.flush()

    def write_data(self, step, pose, time=None, extra_values=None):
        """
        Write down the data to the score file.
        """
        # Create values list
        values = [str('%s' % step)] # Append relax cycle
        values.append(str('%.4f' % self.sfxn(pose))) # append total score
        values.append(str('%.4f' % calculateBindingScore(pose,self.peptide_chain))) # append binding score

        # Append energies by score term
        energies = pose.energies().active_total_energies()
        for score in self.sfxn.get_nonzero_weighted_scoretypes():
            values.append(str('%.4f' % (energies[score.name]*self.sfxn.get_weight(score))))

        # Append time
        if self.time:
            values.append(str('%.2f ' % (time/60.0)))

        if extra_values != None:
            if isinstance(extra_values, dict):
                for item in extra_values:
                    value = extra_values[item]
                    if isinstance(value, float):
                        values.append(str('%.4f' % value))
                    else:
                        values.append(value)

            if isinstance(extra_values, list) or isinstance(extra_values, tuple):
                for value in extra_values:
                    values.append(value)
                    if isinstance(value, float):
                        values.append(str('%.4f' % value))
                    else:
                        values.append(value)

        self.data.write(self.header.format(*values))
        self.data.flush()

    def open(self, mode):
        self.data = open(self.output_file, mode)

    def close(self):
        self.data.close()


def foldTree(pose,flag,peptide_length=False):

    residues = dict()

    for r in range(1,pose.total_residue()+1):
        c = pose.pdb_info().pose2pdb(r).split()[-1]
        residues.setdefault(c,[])
        residues[c].append(r)

    ft = FoldTree()

    if flag == 'Nt':
        # Add edges and jump
        ft.add_edge(residues[b2m_chain][0], residues[b2m_chain][-1], -1)   # chain B
        ft.add_edge(residues[b2m_chain][-1],residues[mhc_chain][0], 1)      # jump
        ft.add_edge(residues[mhc_chain][0], residues[mhc_chain][-1], -1)   # chain A
        ft.add_edge(residues[mhc_chain][centroid_mhc], residues[peptide_chain][0], 2)      # jump
        ft.add_edge(residues[peptide_chain][0], residues[peptide_chain][-1], -1)   # chain C


    elif flag == 'Ct':
        # Add edges and jump
        ft.add_edge(residues[b2m_chain][0], residues[b2m_chain][-1], -1)   # chain B
        ft.add_edge(residues[b2m_chain][-1],residues[mhc_chain][0], 1)      # jump
        ft.add_edge(residues[mhc_chain][0], residues[mhc_chain][-1], -1)   # chain A
        ft.add_edge(residues[mhc_chain][centroid_mhc], residues[peptide_chain][-1], 2)      # jump
        ft.add_edge(residues[peptide_chain][-1],residues[peptide_chain][0], -1)   # chain C


    elif flag == 'middle':
        # Add edges and jump
        centroid_pep = int(math.trunc(peptide_length)/2)
        ft.add_edge(residues[b2m_chain][0], residues[b2m_chain][-1], -1)   # chain B
        ft.add_edge(residues[b2m_chain][-1],residues[mhc_chain][0], 1)      # jump
        ft.add_edge(residues[mhc_chain][0], residues[mhc_chain][-1], -1)   # chain A
        ft.add_edge(residues[mhc_chain][centroid_mhc], residues[peptide_chain][centroid_pep] , 2)      # jump
        ft.add_edge(residues[peptide_chain][centroid_pep], residues[peptide_chain][-1], -1)   # chain C
        ft.add_edge(residues[peptide_chain][centroid_pep], residues[peptide_chain][0], -1)   # chain C

    return ft


def initialMinimization(pose,steps,E_threshold):
    """
    Energy minimization of a given pose with Fast Relax.
    Applied before the MonteCarlo simulation.

    PARAMETERS
    ------------------------------------------------------------
    · pose: PyRosetta pose of the ternary complex.
    · steps: number of minimization steps (type: int).
    · E_threshold: energy threshold to consider if energy has converged.
    ------------------------------------------------------------
    """
    print("""\n#############################
            \nSTART OF INITIAL MINIMIZATION
            \n##############################\n""")

    E_initial = sfxn(pose)
    t_initial = time.time()
    steps += 1

    for s in range(1, steps):
        fast_relax_mover_sampling.apply(pose)
        E_final = sfxn(pose) # ref2015
        dE = E_final-E_initial # energy difference
        t_final = time.time() - t_initial
        if save_pdb:
            if not os.path.exists('./pdb/'):
                os.mkdir('./pdb/')
            pdb_file_name='./pdb/'+name_input+'_minimization_step_'+str(s)+'.pdb'
            with open(pdb_file_name,'w') as pdb:
                pose.dump_pdb(pdb_file_name)
        print("""\n Step %s. Details: \n
                            · Final energy = %.4f
                            · Delta energy = %.4f
                            · Total number of steps = %s \n""" % (s,E_final,dE,minimization_steps))
        if abs(dE) < abs(E_threshold):
                print("""ATTENTION! Energy convergence achieved before reaching total step limit. Energy difference (%.1f) lower than threshold (%.1f)"""%(dE,energy_threshold))
                break
        E_initial = E_final # save energy for next iteration

    print("""\n#################################
            \nEND OF INITIAL MINIMIZATION
            \n#################################\n""")


def applyMovers(c_pose):
    """
    Apply movers to clone pose. First of all it applies either Small or Shear
    mover by performing a random choice. Then it applies the Fast Relax mover.

    PARAMETERS
    ------------------------------------------------------------
    · c_pose: PyRosetta clone pose of the ternary complex.
    ------------------------------------------------------------
    """

    backbone_mover = random.choice([small_mover_sampling, shear_mover_sampling])
    backbone_mover.apply(c_pose)
    fast_relax_mover_sampling.apply(c_pose)


def samplingStep(pose,T,acceptance_ratio):
    """
    Monte Carlo simulation step, with Small and Shear perturbations and then Fast Relax.
    It evaluates if the sampling step is accepted or rejected following the Metropolis criterion and
    taking into account the acceptance ratio.

    PARAMETERS
    ------------------------------------------------------------
    · pose: PyRosetta pose of the ternary complex.
    · T: temperature (type: float).
    · acceptance_ratio: ratio of accepted steps.
    ------------------------------------------------------------
    """

    clone = Pose()
    clone.assign(pose)

    E_initial = sfxn(pose)

    applyMovers(clone)

    E_final = sfxn(clone)

    ratio = np.exp(-(E_final-E_initial)/T)

    probability_move = np.min([ratio, 1])
    random_number = np.random.uniform(0, 1) # normal random number

    if random_number <= probability_move:
        pose.assign(clone)    # modify pose as the step has been accepted
        return 1, E_final
    elif acceptance_ratio < 0.5:
        pose.assign(clone)    # modify pose as the step has been accepted due to low acceptance ratio
        return 1, E_final
    else:
        return 0, E_initial


def samplingMC(pose,pose_reference,description,trees,steps,T):
    """
    Monte Carlo sampling. Applies a random type of Fold Tree at each step (without repeating twice in a row),
    followed by Small and Shear perturbations, and finally a Fast Relax cycle.

    PARAMETERS
    ------------------------------------------------------------
    · pose: PyRosetta pose of the ternary complex.
    · description: name of the original pdb file (type: int).
    · trees: all Fold Tree combinations, Nt/Ct/middle (type: dict)
    · steps: number of sampling steps (type: int).
    · T: temperature (type: float).
    ------------------------------------------------------------
    """
    print("""\n#################################
            \nSTART OF MONTE CARLO SIMULATION
            \n#################################\n""")

    t_initial = time.time()
    steps += 1
    count = 0
    acceptance_ratio = 0

    if pymol:
        pymol.apply(pose)

    old_random_tree = ''
    for s in range(1, steps):

        random_tree = random.choice([k for k in trees.keys()])
        while random_tree == old_random_tree:
            random_tree = random.choice([k for k in trees.keys()])
        old_random_tree = random_tree

        pose.fold_tree(trees[random_tree])
        accept_step, energy = samplingStep(pose,T,acceptance_ratio)

        if accept_step==1:
            t_final = time.time() - t_initial
            pose_rmsd = CA_rmsd(pose, pose_reference)
            peptide_rmsd = peptideRMSD(pose,pose_reference,peptide_chain)
            # my_rmsd = my_RMSD(pose,pose_reference)
            # extra_information = {1:description, 2:pose_rmsd, 3:peptide_rmsd, 4:my_rmsd}
            extra_information = {1:description, 2:pose_rmsd, 3:peptide_rmsd}
            report_data.write_data(s, pose, time=t_final, extra_values=extra_information)
            count+=1
            if pymol:
                pymol.apply(pose)
            if save_pdb:
                if not os.path.exists('./pdb/'):
                    os.mkdir('./pdb/')
                pdb_file_name = './pdb/'+name_input+'_montecarlo_step_'+str(s)+'.pdb'
                with open(pdb_file_name,'w') as pdb:
                    pose.dump_pdb(pdb_file_name)
                print('\nPDB file saved:',pdb_file_name)
            print("""\n Step %s . Details: \n
                            · energy = %.4f
                            · acceptance_ratio = %.2f
                            · accept_step = %s
                            · Fold Tree type = %s
                            · Total number of steps = %s \n""" % (s,energy,acceptance_ratio,accept_step,random_tree,montecarlo_steps))
        else:
            print("\n Step rejected.\n")

        acceptance_ratio = count/s

    if acceptance_ratio < 0.5:
        print("\n WARNING!!! Final acceptance ratio lower than 0.5 WARNING!!!\n")

    print("""\n#################################
            \nEND OF MONTE CARLO SIMULATION
            \n#################################\n""")

    return acceptance_ratio


if __name__ == "__main__":

    # Get arguments
    arguments = getArgs()
    b2m_chain = arguments.b2m_chain
    centroid_mhc = arguments.centroid_mhc
    displace_peptide = arguments.displace_peptide
    energy_threshold = arguments.energy_threshold
    fastrelax_repeats = arguments.fastrelax_repeats
    input_complex = arguments.input_complex
    mhc_chain = arguments.mhc_chain
    minimization_steps = arguments.minimization_steps
    montecarlo_steps = arguments.montecarlo_steps
    neighbours_distance = arguments.neighbours_distance
    peptide_chain = (arguments.peptide_chain).upper()
    peptide_sequence = arguments.peptide_sequence
    if peptide_sequence:
        peptide_sequence = (peptide_sequence).upper()
    pymol = arguments.pymol
    run_id = arguments.run_id
    save_pdb = arguments.save_pdb
    score_file = arguments.score_file
    score_function = arguments.score_function
    shear_angle_helical = arguments.shear_angle_helical
    shear_angle_sheet = arguments.shear_angle_sheet
    shear_angle_loop = arguments.shear_angle_loop
    small_angle_helical = arguments.small_angle_helical
    small_angle_sheet = arguments.small_angle_sheet
    small_angle_loop = arguments.small_angle_loop
    shear_moves = arguments.shear_moves
    small_moves = arguments.small_moves
    temperature = arguments.temperature

    print('\nMAIN PROGRAM EXECUTION HAS STARTED. Run number: %s\n'%(run_id))

    t_initial_program = time.time()

    if input_complex == None:
        raise SystemExit('Please provide the path of the input pdb file.')
    if peptide_chain == None:
        raise SystemExit('Please specify the chain corresponding to the peptide.')

    if not os.path.exists('./tmp/'):
        os.mkdir('./tmp/')

    if not os.path.exists('./output/'):
        os.mkdir('./output/')

    name_input = input_complex.split('/')[-1][:8]

    # Generate clean pdb file without heteroatoms
    cleanATOM(input_complex,out_file='./tmp/clean.pdb')

    # Generate a pose from the clean pdb file
    pose_input = pyrosetta.pose_from_pdb('./tmp/clean.pdb')

    # Generate a clone pose:
    pose_output = pose_input.clone()

    if displace_peptide!=0:
        pose_output = displacePeptide(pose_output,peptide_chain,displace_peptide)
        pdb_peptide_displaced_file_name = '../../../../input/peptide_displaced.pdb'
        with open(pdb_peptide_displaced_file_name,'w') as pdb:
            pose_output.dump_pdb(pdb_peptide_displaced_file_name)
        print('\nPDB file with peptide displaced saved:',pdb_peptide_displaced_file_name)

    if peptide_sequence:
        # Evaluate sequence (length and if it's amino acid sequence), store in variable that will be use to get anchoring residue for Fold Tree
        length_peptide = evaluateSequence(pose_output,peptide_chain,peptide_sequence)
        # Mutate peptide to sequence given by the user
        mutations_peptide, pose_output = mutatePeptide(pose_output, peptide_sequence, peptide_chain)
        print('\nPeptide mutations:',mutations_peptide,'\n')

        if save_pdb:
            if not os.path.exists('./pdb/'):
                os.mkdir('./pdb/')
            pdb_file_name = './pdb/1_'+name_input+'_peptide_mutated_to_'+peptide_sequence+'.pdb'
            with open(pdb_file_name,'w') as pdb:
                pose_output.dump_pdb(pdb_file_name)
            print('\nPDB file saved:',pdb_file_name)
    else:
        print('\nPeptide not mutated.')
        length_peptide = evaluateSequence(pose_output,peptide_chain)

    #########################################################################
    ############################## SAMPLING #################################

    # Generate a clone pose:
    pose_minimization = pose_output.clone()

    # Define Chain Selector
    peptide_selector = ChainSelector(peptide_chain)

    # Define Neighbour Selector (peptide neighbours)
    peptide_neighbours_selector = NeighborhoodResidueSelector()
    peptide_neighbours_selector.set_focus_selector(peptide_selector)
    peptide_neighbours_selector.set_distance(neighbours_distance)  # canvi per 12

    # Define Move Map for Fast Relax (peptide + neighbours)
    ## Enable action, turn on torsion (1)
    enable_torsion = rosetta.core.select.movemap.move_map_action(1)
    ## Deactivate backbone except for the selected chain during relax + neighbours
    move_map_factory_relax = rosetta.core.select.movemap.MoveMapFactory()
    move_map_factory_relax.all_bb(False)
    move_map_factory_relax.add_bb_action(enable_torsion, peptide_selector)
    move_map_factory_relax.add_bb_action(enable_torsion, peptide_neighbours_selector)
    ## Deactivate side-chain except for the selected chain during relax + neighbours
    move_map_factory_relax.all_chi(False)
    move_map_factory_relax.add_chi_action(enable_torsion, peptide_selector)
    move_map_factory_relax.add_chi_action(enable_torsion, peptide_neighbours_selector)

    # Define Move Map for Small and Shear (peptide)
    move_map_factory_sns = rosetta.core.select.movemap.MoveMapFactory()
    move_map_factory_sns.all_bb(False)
    move_map_factory_sns.add_bb_action(enable_torsion, peptide_selector)

    # Define Residue Level Tasks (RLT)
    ## Fix side chains in order to prevent repacking of residues
    prevent_repacking = rosetta.core.pack.task.operation.PreventRepackingRLT()
    ## Movable side chains and fixed sequence in order to only repack residues
    restrict_repacking = rosetta.core.pack.task.operation.RestrictToRepackingRLT()
    ## Prevent repacking of everything except the peptide (flip_subset = True)
    prevent_subset_repacking = rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking, peptide_selector, flip_subset=True)
    ## Allow repacking of peptide and neighbours
    restrict_subset_to_repacking_for_sampling = rosetta.core.pack.task.operation.OperateOnResidueSubset(restrict_repacking, peptide_selector)

    #  Define Task Operation - Sampling
    task_factory_sampling = rosetta.core.pack.task.TaskFactory()
    task_factory_sampling.push_back(prevent_subset_repacking)
    task_factory_sampling.push_back(restrict_subset_to_repacking_for_sampling) # yes side-chains

    # Define Mover
    ## Fast Relax mover for sampling
    fast_relax_mover_sampling = rosetta.protocols.relax.FastRelax(standard_repeats=fastrelax_repeats)
    sfxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function(score_function)
    fast_relax_mover_sampling.set_scorefxn(sfxn)
    fast_relax_mover_sampling.set_movemap_factory(move_map_factory_relax)
    fast_relax_mover_sampling.set_task_factory(task_factory_sampling)
    ## Small mover for sampling
    small_mover_sampling = rosetta.protocols.simple_moves.SmallMover()
    small_mover_sampling.temperature(temperature)
    small_mover_sampling.angle_max("H",small_angle_helical)
    small_mover_sampling.angle_max("E",small_angle_sheet)
    small_mover_sampling.angle_max("L",small_angle_loop)
    small_mover_sampling.nmoves(small_moves)
    # small_mover_sampling.movemap_factory(move_map_factory_sns)
    small_mover_sampling.movemap_factory(move_map_factory_sns)

    ## Shear mover for sampling
    shear_mover_sampling = rosetta.protocols.simple_moves.ShearMover()
    shear_mover_sampling.temperature(temperature)
    shear_mover_sampling.angle_max("H",shear_angle_helical)
    shear_mover_sampling.angle_max("E",shear_angle_sheet)
    shear_mover_sampling.angle_max("L",shear_angle_loop)
    shear_mover_sampling.nmoves(shear_moves)
    # shear_mover_sampling.movemap_factory(move_map_factory_sns)
    shear_mover_sampling.movemap_factory(move_map_factory_sns)

    # Initialize pymol and keep history
    if pymol:
        pymol = rosetta.protocols.moves.PyMOLMover()
        pymol.keep_history(True)

    # Energy of the initial conformation
    E_initial = sfxn(pose_minimization) # ref2015

    # Generate a Data Report
    energies_file = 'output/mc_'+score_file[:-3] + '.png'
    score_file = 'output/'+score_file
    report_data = dataReporter(score_file, sfxn, peptide_chain, time=True)
    report_data.check_previous_run(0, overwrite=True)

    # Minimization of the structure before sampling
    initialMinimization(pose_minimization,minimization_steps,energy_threshold)

    E_minimization = sfxn(pose_minimization)

    # Generate a clone pose:
    pose_sampling = pose_minimization.clone()

    # Fold Tree
    trees = dict()

    trees['Nt'] = foldTree(pose_sampling,'Nt')
    trees['Ct'] = foldTree(pose_sampling,'Ct')
    trees['middle'] = foldTree(pose_sampling,'middle',length_peptide)

    # Monte Carlo
    acc_ratio= samplingMC(pose_sampling,
               pose_output,
               name_input,
               trees,
               montecarlo_steps,
               temperature)

    E_sampling = sfxn(pose_sampling)

    shutil.rmtree('./tmp/')
    #########################################################################
    #########################################################################

    t_final_program = time.time()
    t_execution = t_final_program - t_initial_program
    E_difference = E_initial-E_sampling

    print("""MAIN PROGRAM EXECUTION HAS ENDED. \n
             Summary:\n
             · Total execution time: %.2f\n
             · Initial energy: %.2f\n
             · Energy after %s  minimization steps: %.2f\n
             · Energy after %s  sampling steps: %.2f\n
             · Energy difference: %.2f \n
             · Final acceptance ratio: %.2f\n"""%(t_execution,E_initial,minimization_steps,E_minimization,montecarlo_steps,E_sampling,E_difference,acc_ratio))
