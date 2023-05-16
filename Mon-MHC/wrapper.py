#!/usr/bin/env python3
# coding: utf-8

# WRAPPER
# Development of an MHC-I-peptide binding predictor using Monte Carlo simulations (Vallejo-Vallés et al. 2023)

import io
import os
import re
import shutil
import subprocess
import uuid

import numpy as np

from Bio import SeqIO, AlignIO
from Bio.PDB.Polypeptide import three_to_one
from pkg_resources import resource_stream, Requirement
from scipy.spatial import distance

from pyrosetta import init, get_fa_scorefxn, pose_from_file, pose_from_pdb

init('-mute all')


def copyScript(path, script_to_copy):
    """
    Copy a script from the mhc_sampling package to a given path.

    · path: output path, where the script will be copied (type:str).
    · script_to_copy: script from the mhc_sampling package that will be copied (type:str)
    """

    script = resource_stream(Requirement.parse('mhc_program'),'mhc_program/scripts'+'/'+script_to_copy)

    script = io.TextIOWrapper(script)

    output_path = path+'/'+script_to_copy

    with open(output_path, 'w') as outfile:
        for line in script:
            outfile.write(line)


def writeFastaFile(sequences, output_file):
    """
    Write sequences to a fasta file.

    Parameters
    ----------
    sequences : dict
        Dictionary containing as values the strings representing the sequences
        of the proteins to align and their identifiers as keys.

    output_file : str
        Path to the output fasta file
    """

    # Write fasta file containing the sequences
    with open(output_file, 'w') as of:
        for name in sequences:
            of.write('>'+name+'\n')
            of.write(sequences[name]+'\n')


class mafft:
    """
    Class to hold methods to work with mafft executable.

    Methods
    -------
    multipleSequenceAlignment()
        Execute a multiple sequence alignment of the input sequences
    """

    def multipleSequenceAlignment(sequences, output=None, anysymbol=False, stdout=True, stderr=True,quiet=False):
        """
        Use the mafft executable to perform a multiple sequence alignment.

        Parameters
        ----------
        sequences : dict
            Dictionary containing as values the strings representing the sequences
            of the proteins to align and their identifiers as keys.
        output : str
            File name to write the fasta formatted alignment output.
        anysymbol : bool
            Use unusual symbols in the alignment.
        quiet : bool
            Do not display progress messages.

        Returns
        -------
        alignment : Bio.AlignIO
            Multiple sequence alignment in Biopython format.
        """

        # Generate random variable names
        target_file = '.'+str(uuid.uuid4())+'.fasta.tmp'
        output_file = '.'+str(uuid.uuid4())+'.out.tmp'

        # Write input file containing the sequences
        writeFastaFile(sequences, target_file)

        # Manage stdout and stderr
        if stdout:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        if stderr:
            stderr = None
        else:
            stderr = subprocess.DEVNULL

        # Calculate alignment
        command = 'mafft --auto'
        if anysymbol:
            command += ' --anysymbol'
        if quiet:
            command += ' --quiet'
        command += ' '+target_file+' > '+output_file
        subprocess.run(command, shell=True, stdout=stdout, stderr=stderr)

        # Read aligned file
        alignment = AlignIO.read(output_file, "fasta")

        # Remove temporary file
        os.remove(target_file)
        if output != None:
            shutil.copyfile(output_file, output)
        os.remove(output_file)

        return alignment


def chainSequence(pose,chain):

    """
    This function returns the aminoacid sequence from one chain (one-letter code),

    PARAMETERS
    ------------------------------------------------------------
    · pose: PyRosetta pose of the ternary complex.
    · chain: chain identifier (type: str).
    ------------------------------------------------------------
    """

    sequence = ''

    for index,residue in enumerate(pose.residues):
        # Get chain identifier from pdb
        c = pose.pdb_info().pose2pdb(index+1).split()[-1]
        if c == chain:
            sequence += three_to_one(residue.name()[:3])

    return sequence


def getAlignedResidues(target,domain):

    """
    Returns start and ending residue positions of a given target sequence
    that align with a certain domain of interest ofthe consensus sequence.

    PARAMETERS
    ------------------------------------------------------------
    · target: sequence that will be aligned to the consensus sequence
    · domain: region of interest. Either alpha1, alpha2 or alpha3
    ------------------------------------------------------------
    """

    consensus = 'MAVMAPRTLLLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDQETRNVKAQSQTDRVDLGTLRGYYNQSEAGSHTIQIMYGCDVGSDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAAHEAEQLRAYLDGTCVEWLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWELSSQPTIPIVGIIAGLVLLGAVITGAVVAAVMWRRKSSDRKGGSYTQAASSDSAQGSDVSLTACKV'

    sequences = dict()
    sequences['consensus'] = consensus
    sequences['target'] = target

    # Determine the span of the domain of interest, it varies depending on the MHC.
    # Span information from uniprot.
    if domain == 'alpha1':
        consensus_span = [24,113]
    elif domain == 'alpha2':
        consensus_span = [114,205]
    elif domain == 'alpha3':
        consensus_span = [206,297]

    # MAFFT alignment:
    # mafft_alignment = bioprospecting.alignment.mafft.multipleSequenceAlignment(sequences,quiet=True)
    mafft_alignment = mafft.multipleSequenceAlignment(sequences,quiet=True)

    # Iterate MAFFT alignment to get the indexes of the first and last residue position
    target_residues = []
    for record in mafft_alignment:
        if record.id == 'target':
            domain_of_interest = str(record.seq[consensus_span[0]:consensus_span[1]])
            domain_of_interest= re.sub('-', '', domain_of_interest)
            indexes = [[index.start(0), index.end(0)] for index in re.finditer(domain_of_interest, target)]

    return indexes


def obtainCoordinates(pose, chain, region=None):

    """
    Returns a matrix with the x,y,z backbone coordinates from one chain of a pose.

    PARAMETERS
    ------------------------------------------------------------
    · pose: pose object
    · region: selection of aminoacids to obtain its coordinates, list with the first and last position.
              None as default, it will take into account the whole chain.
    · chain: chain identifier. Chain 1 as default.
    ------------------------------------------------------------
    """

    coordinates = []
    coordinates_index = []

    if region == None:
        region_range = range(0, len(pose.residues)+1)
    else:
        region_range = range(region[0], region[1]+1)

    for index,residue in enumerate(pose.residues):
        c = pose.pdb_info().pose2pdb(index+1).split()[-1]
        if c == chain:
            if index+1 in region_range:
                for atom in residue.all_bb_atoms():
                    atom_coordinates = [residue.xyz(atom)[0],residue.xyz(atom)[1],residue.xyz(atom)[2]]
                    coordinates.append(atom_coordinates)
                    coordinates_index.append(index)

    coordinates = np.array(coordinates)
    coordinates_index = np.array(coordinates_index)

    return coordinates, coordinates_index


def calculateCentroid(pose,chain):

    """
    Returns the centroid (of alpha 1 and 2) coordinates of a given pdb file.

    PARAMETERS
    ------------------------------------------------------------
    · pose: pose object
    ------------------------------------------------------------
    """

    target = chainSequence(pose,chain)

    # Alpha 1
    index = getAlignedResidues(target,domain='alpha1')
    beginning_index = index[0][0]

    # Alpha 2
    index = getAlignedResidues(target,domain='alpha2')
    ending_index = index[0][1]

    span = [beginning_index,ending_index]

    # Obtain coordinates
    coordinates,coordinates_index = obtainCoordinates(pose,chain,region=span)

    # Calculate centroid
    centroid = np.mean(np.asarray(coordinates),axis=0)

    return centroid, coordinates, coordinates_index


def closestResidue(pose,chain):
    """
    Return residue that is closer to given coordinates.

    PARAMETERS
    ------------------------------------------------------------
    · pose: pose object
    · chain: chain of the pose object
    ------------------------------------------------------------
    """

    mhc_centroid, mhc_coordinates, mhc_coordinates_index = calculateCentroid(pose,chain)

    distances = []
    for row in mhc_coordinates:
        distances.append(distance.euclidean(row,mhc_centroid))

    distances = np.array(distances)
    cr_index = np.argmin(distances)
    cr = mhc_coordinates_index[cr_index]

    return cr



class myModels:

    def __init__(self, model_dir):

        # Create class attributes
        self.model_dir = model_dir
        self.model_id = []
        self.poses = {}

        # Load PyRosetta parameters
        self.sfxn = get_fa_scorefxn()

        # Read PDBs as PyRosetta poses
        for model in os.listdir(self.model_dir):
            if model.endswith('.pdb'):
                model_name = model.replace('.pdb', '')
                self.model_id.append(model_name)

                # Read and score pose
                path = self.model_dir+'/'+model
                self.poses[model_name] = pose_from_file(path)
                self.sfxn(self.poses[model_name])

                pose = self.poses[model_name]

        # Sort pdb names for a sorted interation
        self.model_id = sorted(self.model_id)


    def model2pdb(self, pdb_dir):
        """
        Save input models to pdb files.

        PARAMETERS
        ------------------------------------------------------------
        · pdb_dir:  directory to store the output pdb files.
        ------------------------------------------------------------
        """

        # Create output folder
        if not os.path.exists(pdb_dir):
            os.mkdir(pdb_dir)

        # Save model to PDB file
        for model in self.poses:
            path = pdb_dir+'/'+model+'.pdb'
            self.poses[model].dump_file(path)


    def setUpSimulation(self,job_folder,peptide_sequence,replicates=3,energy_threshold=0.5,fastrelax_repeats=1,
                        minimization_steps=5,montecarlo_steps=100,neighbours_distance=7,save_pdb=False,score_function='ref2015',
                        shear_angle_helical=10,shear_angle_sheet=10,shear_angle_loop=10,
                        small_angle_helical=10,small_angle_sheet=10,small_angle_loop=10,
                        shear_moves=5000,small_moves=5000,temperature=0.5,pymol=False,
                        displace_peptide=0):
        """
        Set up the necessary files and directories for the MHC-peptide sampling.
        Returns a list of the jobs that will be given to bsc_calculations.

        PARAMETERS
        ------------------------------------------------------------
        · job_folder: name of the output root directory (type:str).
        · replicates: number of simulation replicates (type:int).
        · peptide_sequence: list of peptide sequences to use for threading (type:list).
        · energy_threshold: energy threshold to consider that energies have converged during minimization (type:float).
        · fastrelax_repeats:
        · minimization_steps: number of steps for the Fast Relax minimization, previous to the Monte Carlo simulation. (type:int).
        · montecarlo_steps: number of steps for Monte Carlo simulation (type:int).
        · neighbours_distance: maximum distance (A) between residues to be considered neighbours (type:float).
        · save_pdb:
        · score_function: score function for the sampling algorithm (type:str).
        · shear_angle: maximum angle allowed for the backbone perturbations of the Shear mover (during Monte Carlo simulation) (type:int).
        · small_angle: maximum angle allowed for the backbone perturbations of the Small mover (during Monte Carlo simulation) (type:int).
        · shear_moves: number of moves for the backbone perturbations of the Shear mover (during Monte Carlo simulation) (type:int).
        · small_moves: number of moves for the backbone perturbations of the Small mover (during Monte Carlo simulation) (type:int).
        · temperature: set up temperature (ºC) for the backbone movers (Small and Shear) and Fast Relax (type:float).
        · pymol: activate pymol (type: bool).
        · displace_peptide
        ------------------------------------------------------------
        """

        replicates += 1

        # Create directories
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if not os.path.exists(job_folder+'/input'):
            os.mkdir(job_folder+'/input')
        if not os.path.exists(job_folder+'/output'):
            os.mkdir(job_folder+'/output')

        path=job_folder+'/input'
        copyScript(path,'main.py')

        self.model2pdb(job_folder+'/input')

        jobs = []
        for m in self.model_id:

            mhc_chain = m[5]
            b2m_chain = m[6]
            peptide_chain = m[7]

            if not os.path.exists(job_folder+'/output/'+m):
                os.mkdir(job_folder+'/output/'+m)

            pose_input = pose_from_pdb(job_folder+'/input/'+m+'.pdb')
            centroid_mhc = closestResidue(pose_input,mhc_chain)

            if peptide_sequence:
                for p in peptide_sequence:

                    if not os.path.exists(job_folder+'/output/'+m+'/'+p):
                        os.mkdir(job_folder+'/output/'+m+'/'+p)

                    for r in range(1,replicates):

                        if not os.path.exists(job_folder+'/output/'+m+'/'+p+'/'+str(r)):
                            os.mkdir(job_folder+'/output/'+m+'/'+p+'/'+str(r))

                        score_file = m+'_'+p+'_'+str(r)+'.sc'

                        # Go to output directory
                        command = 'cd ' + job_folder+'/output/'+m+'/'+p+'/'+ str(r) + '\n'
                        # Execute script
                        command += 'python ../../../../input/main.py'
                        command += ' --input_complex ../../../../input/' + m + '.pdb'
                        command += ' --b2m_chain ' + b2m_chain
                        command += ' --centroid_mhc ' + str(centroid_mhc)
                        command += ' --energy_threshold ' + str(energy_threshold)
                        command += ' --fastrelax_repeats ' + str(fastrelax_repeats)
                        command += ' --mhc_chain '+ mhc_chain
                        command += ' --minimization_steps '+ str(minimization_steps)
                        command += ' --montecarlo_steps ' + str(montecarlo_steps)
                        command += ' --neighbours_distance ' + str(neighbours_distance)
                        command += ' --peptide_chain '+ peptide_chain
                        command += ' --peptide_sequence '+ p
                        command += ' --run_id '+ str(r)
                        command += ' --score_file ' + score_file
                        command += ' --score_function ' + score_function
                        command += ' --shear_angle_helical ' + str(shear_angle_helical)
                        command += ' --shear_angle_sheet ' + str(shear_angle_sheet)
                        command += ' --shear_angle_loop ' + str(shear_angle_loop)
                        command += ' --small_angle_helical '+ str(small_angle_helical)
                        command += ' --small_angle_sheet '+ str(small_angle_sheet)
                        command += ' --small_angle_loop '+ str(small_angle_loop)
                        command += ' --shear_moves '+ str(shear_moves)
                        command += ' --small_moves ' + str(small_moves)
                        command += ' --temperature '+ str(temperature)
                        command += ' --displace_peptide '+ str(displace_peptide)
                        if pymol:
                            command += ' --pymol'
                        if save_pdb:
                            command += ' --save_pdb'
                        command += '\n'
                        command += 'cd ../../../\n'
                        jobs.append(command)

            else:
                if not os.path.exists(job_folder+'/output/'+m+'/native_peptide'):
                    os.mkdir(job_folder+'/output/'+m+'/native_peptide')

                for r in range(1,replicates):

                    if not os.path.exists(job_folder+'/output/'+m+'/native_peptide/'+str(r)):
                        os.mkdir(job_folder+'/output/'+m+'/native_peptide/'+str(r))

                    score_file = m+'_native_peptide_'+str(r)+'.sc'

                    # Go to output directory
                    command = 'cd ' + job_folder+'/output/'+m+'/native_peptide/'+ str(r) + '\n'
                    # Execute script
                    command += 'python ../../../../input/main.py'
                    command += ' --input_complex ../../../../input/' + m + '.pdb'
                    command += ' --b2m_chain ' + b2m_chain
                    command += ' --centroid_mhc ' + str(centroid_mhc)
                    command += ' --energy_threshold ' + str(energy_threshold)
                    command += ' --fastrelax_repeats ' + str(fastrelax_repeats)
                    command += ' --mhc_chain '+ mhc_chain
                    command += ' --minimization_steps '+ str(minimization_steps)
                    command += ' --montecarlo_steps ' + str(montecarlo_steps)
                    command += ' --neighbours_distance ' + str(neighbours_distance)
                    command += ' --peptide_chain '+ peptide_chain
                    command += ' --run_id '+ str(r)
                    command += ' --score_file ' + score_file
                    command += ' --score_function ' + score_function
                    command += ' --shear_angle_helical ' + str(shear_angle_helical)
                    command += ' --shear_angle_sheet ' + str(shear_angle_sheet)
                    command += ' --shear_angle_loop ' + str(shear_angle_loop)
                    command += ' --small_angle_helical '+ str(small_angle_helical)
                    command += ' --small_angle_sheet '+ str(small_angle_sheet)
                    command += ' --small_angle_loop '+ str(small_angle_loop)
                    command += ' --shear_moves '+ str(shear_moves)
                    command += ' --small_moves ' + str(small_moves)
                    command += ' --temperature '+ str(temperature)
                    command += ' --displace_peptide '+ str(displace_peptide)
                    if pymol:
                        command += ' --pymol'
                    if save_pdb:
                        command += ' --save_pdb'
                    command += '\n'
                    command += 'cd ../../../../../\n'
                    jobs.append(command)
        return jobs




    def setUpSimulationCase4(self,job_folder,peptide_sequence,replicates=3,energy_threshold=0.5,fastrelax_repeats=1,
                        minimization_steps=5,montecarlo_steps=100,neighbours_distance=7,save_pdb=False,score_function='ref2015',
                        shear_angle_helical=10,shear_angle_sheet=10,shear_angle_loop=10,
                        small_angle_helical=10,small_angle_sheet=10,small_angle_loop=10,
                        shear_moves=5000,small_moves=5000,temperature=0.5,pymol=False,
                        displace_peptide=0):
        """
        Set up the necessary files and directories for the MHC-peptide sampling.
        Returns a list of the jobs that will be given to bsc_calculations.
        Chain ID is different than pdb naming (mhc and b2m are from the centroid pdb)

        PARAMETERS
        ------------------------------------------------------------
        · job_folder: name of the output root directory (type:str).
        · replicates: number of simulation replicates (type:int).
        · peptide_sequence: list of peptide sequences to use for threading (type:list).
        · energy_threshold: energy threshold to consider that energies have converged during minimization (type:float).
        · fastrelax_repeats:
        · minimization_steps: number of steps for the Fast Relax minimization, previous to the Monte Carlo simulation. (type:int).
        · montecarlo_steps: number of steps for Monte Carlo simulation (type:int).
        · neighbours_distance: maximum distance (A) between residues to be considered neighbours (type:float).
        · save_pdb:
        · score_function: score function for the sampling algorithm (type:str).
        · shear_angle: maximum angle allowed for the backbone perturbations of the Shear mover (during Monte Carlo simulation) (type:int).
        · small_angle: maximum angle allowed for the backbone perturbations of the Small mover (during Monte Carlo simulation) (type:int).
        · shear_moves: number of moves for the backbone perturbations of the Shear mover (during Monte Carlo simulation) (type:int).
        · small_moves: number of moves for the backbone perturbations of the Small mover (during Monte Carlo simulation) (type:int).
        · temperature: set up temperature (ºC) for the backbone movers (Small and Shear) and Fast Relax (type:float).
        · pymol: activate pymol (type: bool).
        · displace_peptide
        ------------------------------------------------------------
        """

        replicates += 1

        # Create directories
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if not os.path.exists(job_folder+'/input'):
            os.mkdir(job_folder+'/input')
        if not os.path.exists(job_folder+'/output'):
            os.mkdir(job_folder+'/output')

        path=job_folder+'/input'
        copyScript(path,'main.py')

        self.model2pdb(job_folder+'/input')

        jobs = []
        for m in self.model_id:

            mhc_chain = 'A'
            b2m_chain = 'B'
            peptide_chain = 'C'

            if not os.path.exists(job_folder+'/output/'+m):
                os.mkdir(job_folder+'/output/'+m)

            pose_input = pose_from_pdb(job_folder+'/input/'+m+'.pdb')
            centroid_mhc = closestResidue(pose_input,mhc_chain)

            if peptide_sequence:
                for p in peptide_sequence:

                    if not os.path.exists(job_folder+'/output/'+m+'/'+p):
                        os.mkdir(job_folder+'/output/'+m+'/'+p)

                    for r in range(1,replicates):

                        if not os.path.exists(job_folder+'/output/'+m+'/'+p+'/'+str(r)):
                            os.mkdir(job_folder+'/output/'+m+'/'+p+'/'+str(r))

                        score_file = m+'_'+p+'_'+str(r)+'.sc'

                        # Go to output directory
                        command = 'cd ' + job_folder+'/output/'+m+'/'+p+'/'+ str(r) + '\n'
                        # Execute script
                        command += 'python ../../../../input/main.py'
                        command += ' --input_complex ../../../../input/' + m + '.pdb'
                        command += ' --b2m_chain ' + b2m_chain
                        command += ' --centroid_mhc ' + str(centroid_mhc)
                        command += ' --energy_threshold ' + str(energy_threshold)
                        command += ' --fastrelax_repeats ' + str(fastrelax_repeats)
                        command += ' --mhc_chain '+ mhc_chain
                        command += ' --minimization_steps '+ str(minimization_steps)
                        command += ' --montecarlo_steps ' + str(montecarlo_steps)
                        command += ' --neighbours_distance ' + str(neighbours_distance)
                        command += ' --peptide_chain '+ peptide_chain
                        command += ' --peptide_sequence '+ p
                        command += ' --run_id '+ str(r)
                        command += ' --score_file ' + score_file
                        command += ' --score_function ' + score_function
                        command += ' --shear_angle_helical ' + str(shear_angle_helical)
                        command += ' --shear_angle_sheet ' + str(shear_angle_sheet)
                        command += ' --shear_angle_loop ' + str(shear_angle_loop)
                        command += ' --small_angle_helical '+ str(small_angle_helical)
                        command += ' --small_angle_sheet '+ str(small_angle_sheet)
                        command += ' --small_angle_loop '+ str(small_angle_loop)
                        command += ' --shear_moves '+ str(shear_moves)
                        command += ' --small_moves ' + str(small_moves)
                        command += ' --temperature '+ str(temperature)
                        command += ' --displace_peptide '+ str(displace_peptide)
                        if pymol:
                            command += ' --pymol'
                        if save_pdb:
                            command += ' --save_pdb'
                        command += '\n'
                        command += 'cd ../../../\n'
                        jobs.append(command)

            else:
                if not os.path.exists(job_folder+'/output/'+m+'/native_peptide'):
                    os.mkdir(job_folder+'/output/'+m+'/native_peptide')

                for r in range(1,replicates):

                    if not os.path.exists(job_folder+'/output/'+m+'/native_peptide/'+str(r)):
                        os.mkdir(job_folder+'/output/'+m+'/native_peptide/'+str(r))

                    score_file = m+'_native_peptide_'+str(r)+'.sc'

                    # Go to output directory
                    command = 'cd ' + job_folder+'/output/'+m+'/native_peptide/'+ str(r) + '\n'
                    # Execute script
                    command += 'python ../../../../input/main.py'
                    command += ' --input_complex ../../../../input/' + m + '.pdb'
                    command += ' --b2m_chain ' + b2m_chain
                    command += ' --centroid_mhc ' + str(centroid_mhc)
                    command += ' --energy_threshold ' + str(energy_threshold)
                    command += ' --fastrelax_repeats ' + str(fastrelax_repeats)
                    command += ' --mhc_chain '+ mhc_chain
                    command += ' --minimization_steps '+ str(minimization_steps)
                    command += ' --montecarlo_steps ' + str(montecarlo_steps)
                    command += ' --neighbours_distance ' + str(neighbours_distance)
                    command += ' --peptide_chain '+ peptide_chain
                    command += ' --run_id '+ str(r)
                    command += ' --score_file ' + score_file
                    command += ' --score_function ' + score_function
                    command += ' --shear_angle_helical ' + str(shear_angle_helical)
                    command += ' --shear_angle_sheet ' + str(shear_angle_sheet)
                    command += ' --shear_angle_loop ' + str(shear_angle_loop)
                    command += ' --small_angle_helical '+ str(small_angle_helical)
                    command += ' --small_angle_sheet '+ str(small_angle_sheet)
                    command += ' --small_angle_loop '+ str(small_angle_loop)
                    command += ' --shear_moves '+ str(shear_moves)
                    command += ' --small_moves ' + str(small_moves)
                    command += ' --temperature '+ str(temperature)
                    command += ' --displace_peptide '+ str(displace_peptide)
                    if pymol:
                        command += ' --pymol'
                    if save_pdb:
                        command += ' --save_pdb'
                    command += '\n'
                    command += 'cd ../../../../../\n'
                    jobs.append(command)
        return jobs
