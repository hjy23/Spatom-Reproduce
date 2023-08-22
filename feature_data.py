# from Bio.PDB import *
import random

import freesasa
import os
import pickle
import itertools
from Bio.PDB import PDBParser, PDBIO, Select
import numpy as np
import math
from Bio.PDB.DSSP import DSSP
import glob

with open('./Data/DBD/data/ubToB_A.pkl', 'rb') as f:
    test_list = pickle.load(f)

with open('./Data/DBD/data/train462_list.pkl', 'rb') as f:
    ubToB_list = pickle.load(f)

fasta_dict = pickle.load(open('./Data/DBD/data/seq_single_ub_A.pkl', 'rb'))
# fasta_dict = pickle.load(open('./Data/DBD/data/label_b.pkl', 'rb'))


def onehot_dict(list, seq_dict):
    AA_index = {j: i for i, j in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    onehot_dict = {}
    for i in list:
        protein = i[0:4]
        c = i.split('_')[1].upper()
        l = i.split('_')[2]
        onehot = []
        for ci in c:
            for AA in seq_dict[protein + '_' + ci + '_' + l]:
                zero = [0] * 20
                zero[AA_index[AA]] = 1
                onehot.append(zero)
        onehot_dict[i] = onehot
    f = open('./Data/DBD/data/feature_data/One_hot_dict.pkl', 'wb')
    pickle.dump(onehot_dict, f)


def PSSM_dict(data_list, PSSM_dir='./Data/DBD/data/feature/pssm/'):
    PSSM_dict = {}
    # max = np.array(
    #     [7., 9., 9., 9., 12., 9., 8., 8., 12., 8., 7., 9., 11., 10., 9., 8., 8., 13., 10., 8.])
    # min = np.array(
    #     [-10., - 12., - 12., - 12., - 11., - 11., - 12., - 12., - 12., - 12., - 12., - 12., - 11., - 11.,
    #      - 12., - 11., - 10., - 12., - 11., - 11.])

    max = np.array([8., 9., 9., 9., 12., 9., 8., 8., 12., 9., 7., 8., 11.,
           10., 9., 8., 8., 13., 10., 8.])

    min = np.array([-10., -11., -12., -12., -11., -10., -11., -11., -11., -10., -11.,
           -11., -10., -11., -12., -11., -10., -11., -10., -11.])
    for i in data_list:
        protein = i[0:4]
        c = i.split('_')[1].upper()
        l = i.split('_')[2]
        PSSM_matrix = []
        for ci in c:
            pdbid = f'{protein}_{ci}_{l}'
            try:
                PSSM_file = open(f'{PSSM_dir}{pdbid}.pssm', 'r')
                for line in PSSM_file:
                    if line != None and len(line.split()) > 40:
                        PSSM_line = line.split()[2:22]
                        PSSM_line = list(map(float, PSSM_line))
                        PSSM_line = ((np.array(PSSM_line) - min) / (max - min)).tolist()
                        PSSM_matrix.append(PSSM_line)

            except:
                # PSSM_matrix = []
                for j in range(len(fasta_dict[pdbid])):
                    PSSM_line = [0.0]*20
                    PSSM_matrix.append(PSSM_line)
                # PSSM_dict[pdbid] = PSSM_matrix
        PSSM_dict[i] = PSSM_matrix
    f = open('./Data/DBD/data/feature_data/PSSM_dict.pkl', 'wb')
    pickle.dump(PSSM_dict, f)


def HMM_dict(seq_list, hmm_dir, feature_dir):
    hmm_dict = {}
    for seqid in seq_list:
        pdbid = seqid[:4]
        chian = seqid[5:]
        hmm_feature = []
        for c in chian:
            file = pdbid + '_' + c + '.hhm'
            with open(hmm_dir + file, 'r') as fin:
                fin_data = fin.readlines()
                hhm_begin_line = 0
                hhm_end_line = 0
                for i in range(len(fin_data)):
                    if '#' in fin_data[i]:
                        hhm_begin_line = i + 5
                    elif '//' in fin_data[i]:
                        hhm_end_line = i
                feature = np.zeros([int((hhm_end_line - hhm_begin_line) / 3), 20])
                axis_x = 0
                for i in range(hhm_begin_line, hhm_end_line, 3):
                    line1 = fin_data[i].split()[2:-1]
                    line2 = fin_data[i + 1].split()
                    axis_y = 0
                    for j in line1:
                        if j == '*':
                            feature[axis_x][axis_y] = 9999 / 10000.0
                        else:
                            feature[axis_x][axis_y] = float(j) / 10000.0
                        axis_y += 1
                    # for j in line2:
                    #     if j == '*':
                    #         feature[axis_x][axis_y] = 9999 / 10000.0
                    #     else:
                    #         feature[axis_x][axis_y] = float(j) / 10000.0
                    #     axis_y += 1
                    axis_x += 1
                feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
                hmm_feature.append(feature)
        hmm_dict[seqid] = np.concatenate(hmm_feature, dim=0)
    with open(feature_dir + 'HMM_dict.pkl', 'wb') as f:
        pickle.dump(hmm_dict, f)
    return


def RSA_dict(data_list):
    RSA_dict = {}
    for i in data_list:
        protein = i[0:4]
        c = i.split('_')[1].upper()
        l = i.split('_')[2]

        structure = freesasa.Structure(f'./Data/DBD/data/ub_pdb/{i}.pdb')
        result = freesasa.calc(structure, freesasa.Parameters(
            {'algorithm': freesasa.LeeRichards, 'n-slices': 100, 'probe-radius': 1.4}))
        residueAreas = result.residueAreas()
        RSA = []
        for ci in c:
            pdbid = f'{protein}_{ci}_{l}'
            if ci == '*':
                ci = ' '
            for r in residueAreas[ci].keys():
                RSA_AA = []
                RSA_AA.append(min(1, residueAreas[ci][r].relativeTotal))
                RSA_AA.append(min(1, residueAreas[ci][r].relativePolar))
                RSA_AA.append(min(1, residueAreas[ci][r].relativeApolar))
                RSA_AA.append(min(1, residueAreas[ci][r].relativeMainChain))
                if math.isnan(residueAreas[ci][r].relativeSideChain):
                    RSA_AA.append(0)
                else:
                    RSA_AA.append(min(1, residueAreas[ci][r].relativeSideChain))
                RSA.append(RSA_AA)
        RSA_dict[i] = RSA
    f = open('./Data/DBD/data/feature_data/RSA_dict.pkl', 'wb')
    pickle.dump(RSA_dict, f)


class ChianSelect(Select):
    def __init__(self, chain_letter):
        self.chain_letter = chain_letter

    def accept_chain(self, chain):
        if chain.get_id() in self.chain_letter:
            return True
        else:
            return False


def Single_PDB(data_list):
    for i in data_list:
        protein = i[0:4]
        c = i[5:].upper()
        p = PDBParser(QUIET=1)
        pdb = p.get_structure(protein, "./Data/test34/" + i + '.pdb')
        io = PDBIO()
        io.set_structure(pdb)
        chain_list = [c]
        for ci in c:
            chain = [ci]
            io.save('./Data/single_pdb/' + protein + '_' + ci + '.pdb', ChianSelect(chain))


class CA_Select(Select):
    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " else 0

    def accept_atom(self, atom):
        if atom.get_name() == 'CA':
            return True
        else:
            return False


def CA_PDB(data_list):
    for i in data_list:
        protein = i[0:4]
        c = i[5:].upper()
        p = PDBParser(QUIET=1)
        for ci in c:
            pdb = p.get_structure(protein, "./Data/single_pdb/" + i + '.pdb')
            io = PDBIO()
            io.set_structure(pdb)
            io.save('./Data/updata/pdb/CA_pdb/' + protein + '_' + ci + '.pdb', CA_Select())


def AA_property(data_list, seq_dict):
    Side_Chain_Atom_num = {'A': 5.0, 'C': 6.0, 'D': 8.0, 'E': 9.0, 'F': 11.0, 'G': 4.0, 'H': 10.0, 'I': 8.0, 'K': 9.0,
                           'L': 8.0, 'M': 8.0, 'N': 8.0, 'P': 7.0, 'Q': 9.0, 'R': 11.0, 'S': 6.0, 'T': 7.0, 'V': 7.0,
                           'W': 14.0, 'Y': 12.0}
    Side_Chain_Charge_num = {'A': 0.0, 'C': 0.0, 'D': -1.0, 'E': -1.0, 'F': 0.0, 'G': 0.0, 'H': 1.0, 'I': 0.0, 'K': 1.0,
                             'L': 0.0, 'M': 0.0, 'N': 0.0, 'P': 0.0, 'Q': 0.0, 'R': 1.0, 'S': 0.0, 'T': 0.0, 'V': 0.0,
                             'W': 0.0, 'Y': 0.0}
    Side_Chain_hydrogen_bond_num = {'A': 2.0, 'C': 2.0, 'D': 4.0, 'E': 4.0, 'F': 2.0, 'G': 2.0, 'H': 4.0, 'I': 2.0,
                                    'K': 2.0, 'L': 2.0, 'M': 2.0, 'N': 4.0, 'P': 2.0, 'Q': 4.0, 'R': 4.0, 'S': 4.0,
                                    'T': 4.0, 'V': 2.0, 'W': 3.0, 'Y': 3.0}
    Side_Chain_pKa = {'A': 7.0, 'C': 7.0, 'D': 3.65, 'E': 3.22, 'F': 7.0, 'G': 7.0, 'H': 6.0, 'I': 7.0, 'K': 10.53,
                      'L': 7.0, 'M': 7.0, 'N': 8.18, 'P': 7.0, 'Q': 7.0, 'R': 12.48, 'S': 7.0, 'T': 7.0, 'V': 7.0,
                      'W': 7.0, 'Y': 10.07}
    Hydrophobicity = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': 3.9,
                      'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
                      'W': -0.9, 'Y': -1.3}

    AA_property_dict = {}
    for i in data_list:
        protein = i[0:4]
        c = i.split('_')[1].upper()
        l = i.split('_')[2]
        AA_protein = []
        for ci in c:
            for AA in seq_dict[protein + '_' + ci + '_' + l]:
                AA_AA = []
                AA_AA.append(Side_Chain_Atom_num[AA])
                AA_AA.append(Side_Chain_Charge_num[AA])
                AA_AA.append(Side_Chain_hydrogen_bond_num[AA])
                AA_AA.append(Side_Chain_pKa[AA])
                AA_AA.append(Hydrophobicity[AA])
                AA_protein.append(AA_AA)
        AA_property_dict[i] = AA_protein
    f = open('./Data/DBD/data/feature_data/AA_property_dict.pkl', 'wb')
    pickle.dump(AA_property_dict, f)


def DSSP_dict(data_list, seq_dict):
    SS_dict = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}
    dssp_dict = {}
    for data in data_list:
        protein = data[0:4]
        c = data.split('_')[1].upper()
        l = data.split('_')[2]
        dssp_feature = []
        for ci in c:
            pdbid = protein + '_' + ci + '_' + l
            if ci == '*':
                ci = ' '
            ref_seq = seq_dict[pdbid]
            p = PDBParser(QUIET=1)
            structure = p.get_structure(protein, './Data/DBD/data/ub_pdb/' + data + '.pdb')
            model = structure[0]
            dssp = DSSP(model, './Data/DBD/data/ub_pdb/' + data + '.pdb',
                        dssp='/mnt/data0/Hanjy/.conda/envs/pytorch/bin/mkdssp')
            key_list = []
            for i in dssp.keys():
                if i[0] == ci:
                    key_list.append(i)
            dssp_matrix = []
            seq = ""
            for i in key_list:
                SS = dssp[i][2]
                AA = dssp[i][1]
                seq += AA
                phi = dssp[i][4]
                psi = dssp[i][5]
                raw = []
                raw.append(np.sin(phi * (np.pi / 180)))
                raw.append(np.sin(psi * (np.pi / 180)))
                raw.append(np.sin(phi * (np.pi / 180)))
                raw.append(np.cos(psi * (np.pi / 180)))
                ss_raw = [0] * 9
                ss_raw[SS_dict[SS]] = 1
                raw.extend(ss_raw)
                dssp_matrix.append(raw)
            pad = []
            pad.append(np.sin(360 * (np.pi / 180)))
            pad.append(np.sin(360 * (np.pi / 180)))
            pad.append(np.cos(360 * (np.pi / 180)))
            pad.append(np.cos(360 * (np.pi / 180)))
            ss_pad = [0] * 9
            ss_pad[-1] = 1
            pad.extend(ss_pad)
            pad_dssp_matrix = []
            p_ref = 0
            for i in range(len(seq)):
                while p_ref < len(ref_seq) and seq[i] != ref_seq[p_ref]:
                    pad_dssp_matrix.append(pad)
                    p_ref += 1
                if p_ref < len(ref_seq):  # aa matched
                    pad_dssp_matrix.append(dssp_matrix[i])
                    p_ref += 1
            if len(pad_dssp_matrix) != len(ref_seq):
                for i in range(len(ref_seq) - len(pad_dssp_matrix)):
                    pad_dssp_matrix.append(pad)
            dssp_feature.extend(pad_dssp_matrix)
        dssp_dict[data] = dssp_feature
    f = open('./Data/DBD/data/feature_data/DSSP_dict.pkl', 'wb')
    pickle.dump(dssp_dict, f)


def Dist_adj(data_list):
    def dictance(xyz, position):
        xyz = xyz - xyz[position]
        dictance = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2).tolist()
        return dictance

    distance_dict = {}
    for data in data_list:
        protein = data[0:4]
        c = data.split('_')[1].upper()
        l = data.split('_')[2]
        p = PDBParser(QUIET=1)
        structure = p.get_structure(protein, "./Data/DBD/data/CA_pdb/" + data + '.pdb')
        # for ci in c:
            # structure = protein.get_structure(protein, "./case_study/CA_PDB/" + protein + '_' + ci + '.pdb')
        distance_list = []
        chain_list = []
        for ci in c:
            # pdbid = protein + '_' + ci + '_' + l
            if ci == '0':
                ci = ' '
            try:
                for residue in structure[0][ci]:
                    for atom in residue:
                        chain_list.append(atom.get_vector().get_array())
            except:
                for residue in structure[0][' ']:
                    for atom in residue:
                        chain_list.append(atom.get_vector().get_array())
        for i, center in enumerate(chain_list):
            distance_list.append(dictance(chain_list, i))
        distance_dict[data] = distance_list
    f = open('./Data/DBD/data/feature_data/Dist_dict.pkl', 'wb')
    pickle.dump(distance_dict, f)


def CX_DPX(data_list, path="./Data/DBD/data/feature/CX_DPX"):
    CX_DPX_dict = {}
    for i in data_list:
        protein = i[0:4]
        c = i.split('_')[1].upper()
        l = i.split('_')[2]
        # for ci in c:
        if c == '*':
            i = protein + '___' + l

        CX_DPX_protein = []
        try:
            fname = glob.glob(path + "/" + i + "*.tbl")[0]
        except:
            print(i)
        f = open(fname, 'r')
        for line in f:
            if line.split() != []:
                if line.split()[0] in c:
                    AA_CX_DPX = list(map(float, line.split()[3:]))
                    CX_DPX_protein.append(AA_CX_DPX)
        i = protein + '_' + c + '_' + l
        CX_DPX_dict[i] = CX_DPX_protein
    f = open('./Data/DBD/data/feature_data/CX_DPX.pkl', 'wb')
    pickle.dump(CX_DPX_dict, f)



#onehot_dict(test_list, fasta_dict)
#PSSM_dict(test_list)
RSA_dict(test_list)
# Single_PDB(test_list)
# CA_PDB(test_list)
#AA_property(test_list, fasta_dict)
DSSP_dict(test_list, fasta_dict)
# Dist_adj(test_list)

print('done!!')
