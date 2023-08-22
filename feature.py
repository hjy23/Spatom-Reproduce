import torch
from torch_geometric.data import Data
import pickle
import numpy as np
import os


onehot_dict = pickle.load(open('./Data/DBD/data/feature_data/One_hot_dict.pkl', 'rb'))
PSSM_dict = pickle.load(open('./Data/DBD/data/feature_data/PSSM_dict.pkl', 'rb'))
RSA_dict = pickle.load(open('./Data/DBD/data/feature_data/RSA_dict.pkl', 'rb'))
AA_property_dict = pickle.load(open('./Data/DBD/data/feature_data/AA_property_dict.pkl', 'rb'))
DSSP_dict = pickle.load(open('./Data/DBD/data/feature_data/DSSP_dict.pkl', 'rb'))
with open('./Data/DBD/data/feature_data/Dist_dict.pkl', 'rb') as f:
    Dist_dict = pickle.load(f)

train_data = pickle.load(open('./Data/DBD/data/train462_list.pkl', 'rb'))
test_data = pickle.load(open('./Data/DBD/data/test80_list.pkl', 'rb'))
with open('./Data/DBD/data/label_dict_A.pkl', 'rb') as f:
    all_data_label = pickle.load(f)

def normalized_Laplacian(matrix):
    degree_sum = np.array((matrix.sum(1)))
    D_diag = (degree_sum ** -0.5).flatten()
    D_diag[np.isinf(D_diag)] = 0
    D_inv = np.diag(D_diag)
    Lap = D_inv @ matrix @ D_inv
    return Lap

def edge_weight(dist):
    matrix = dist.clone()
    softmax = torch.nn.Softmax(dim=0)
    dist = softmax(1./(torch.log(torch.log(dist+2))))
    #dist = 1. / (torch.log(torch.log(dist + 2)))
    dist[matrix>14] = 0
    return dist

def normalized_Laplacian(matrix):
    degree_sum = np.array((matrix.sum(1)))
    D_diag = (degree_sum ** -0.5).flatten()
    D_diag[np.isinf(D_diag)] = 0
    D_inv = np.diag(D_diag)
    Lap = D_inv @ matrix @ D_inv
    return Lap

def edge_weight(dist):
    matrix = dist.clone()
    softmax = torch.nn.Softmax(dim=0)
    dist = softmax(1./(torch.log(torch.log(dist+2))))
    dist[matrix>14] = 0
    return dist

def feature_Adj(data_list,label,mode):
    Datasets = []
    for i in data_list:
        # protein = '3R9A_A'
        protein = i
        feature = []
        RSA = []
        for k in range(len(onehot_dict[i])):
            AA = []
            AA.extend(onehot_dict[i][k])
            AA.extend(PSSM_dict[i][k])
            AA.extend(RSA_dict[i][k])
            AA.extend(AA_property_dict[i][k])
            AA.extend(DSSP_dict[i][k])
            feature.append(AA)
            RSA.append(RSA_dict[i][k][0])
        if mode == 'test':
            pos = np.where(np.array(RSA) >= 0.05)[0].tolist()
        else:
            pos = np.where(np.array(RSA) >= 0.05)[0].tolist()
        if mode == 'test':
            labels = torch.tensor(np.array(label[protein]), dtype=torch.float)
            Dist = edge_weight(torch.tensor(Dist_dict[i]))[pos, :][:, pos]
            #Dist = torch.tensor(normalized_Laplacian(np.where(np.array(Dist_dict[i]) < 14, 1, 0)))[pos, :][:, pos]
            feature = torch.tensor(np.array(feature)[pos, :], dtype=torch.float)
            adj = torch.tensor(np.where(np.array(Dist_dict[i]) < 14, 1, 0)[pos,:][:,pos])
        else:
            labels = torch.tensor(np.array(label[protein])[pos], dtype=torch.float)
            #labels = torch.tensor(np.array(label[protein]), dtype=torch.float)
            Dist = edge_weight(torch.tensor(Dist_dict[i]))[pos, :][:, pos]
            #Dist = torch.tensor(normalized_Laplacian(np.where(np.array(Dist_dict[i]) < 14, 1, 0)))[pos, :][:, pos]
            feature = torch.tensor(np.array(feature)[pos, :], dtype=torch.float)
            adj = torch.tensor(np.where(np.array(Dist_dict[i]) < 14, 1, 0)[pos,:][:,pos])
        data = Data(x=feature,y=labels)
        data.name = i
        data.dist = Dist
        data.POS = pos
        length = len(label[protein])
        data.length = length
        data.adj = adj

        Datasets.append(data)
    f = open('./Data/DBD/data/feature_extract/'+mode+'_feature.pkl', 'wb')
    pickle.dump(Datasets, f)
    
path_dir = "./Data/DBD/data/feature_extract/"
if not os.path.exists(path_dir):
    os.makedirs(path_dir)
feature_Adj(test_data,all_data_label, 'test')
feature_Adj(train_data,all_data_label,'train')

