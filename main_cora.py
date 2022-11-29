import torch
import csv
import math
import itertools
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model_cora import FT_VGAE
from preprocessing import load_data, sparse_to_tuple, preprocess_graph
from sklearn.metrics import confusion_matrix
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.patches as patches

# Dataset Name
dataset = "Cora"
print("Cora dataset")
adj, features, labels = load_data('cora', './Cora')
nClusters_1 = 7
nClusters_2 = 40
n_neighbors_comp = 3

# Network parameters
alpha = 1.
gamma_1 = 1.
gamma_2 = 1.
num_neurons = 32
embedding_size = 16
save_path = "./results/"

# T1 parameters
epochs_T1 = 60
lr_T1 = 0.001

# T2 parameters
epochs_T2 = 1
lr_T2 = 0.001

# T3 parameters
epochs_T3 = 1
lr_T3 = 0.001

# Some preprocessing
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()
adj_norm = preprocess_graph(adj)
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2])).to("cuda:0")
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]), torch.Size(adj_label[2])).to("cuda:0")
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]), torch.Size(features[2])).to("cuda:0")
weight_mask_orig = adj_label.to_dense().view(-1) == 1
weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
weight_tensor_orig[weight_mask_orig] = pos_weight_orig
weight_tensor_orig = weight_tensor_orig.to("cuda:0")

# Create and train Model
logfile = open(save_path + dataset + '/log_train.csv', 'w')
logwriter = csv.DictWriter(logfile, fieldnames=['n_neighbors_comp', 'nClusters_2', 'acc', 'nmi', 'ari'])
logwriter.writeheader()
epoch_index = 0
for n_neighbors_comp in [3, 5, 7, 9, 11]:                    
    for nClusters_2 in [50, 100, 150, 200, 250]:    
        network = FT_VGAE(n_neighbors_comp=n_neighbors_comp, num_neurons=num_neurons, num_features=num_features, embedding_size=embedding_size, nClusters_1=nClusters_1, nClusters_2=nClusters_2, activation="ReLU", alpha=alpha, gamma_1=gamma_1, gamma_2=gamma_2).to("cuda:0")
        network.load_state_dict(torch.load(save_path + dataset + '/T0/model_best.pk'))
        epoch_index = network.train_phase_1(epoch_index, adj_norm, features, adj_label, labels, weight_tensor_orig, norm, optimizer="Adam", epochs=epochs_T1, lr=lr_T1, save_path=save_path, dataset=dataset)
        epoch_index = epoch_index + 1
        acc_best, nmi_best, ari_best = 0, 0, 0
        for i in range(50):
            acc_best, nmi_best, ari_best, epoch_index = network.train_phase_3_1(acc_best, nmi_best, ari_best, epoch_index, adj_norm, features, adj_label, labels, weight_tensor_orig, norm, optimizer="Adam", epochs=epochs_T2, lr=lr_T2, save_path=save_path, dataset=dataset)
            epoch_index = epoch_index + 1
            acc_best, nmi_best, ari_best, epoch_index = network.train_phase_3_2(acc_best, nmi_best, ari_best, epoch_index, adj_norm, features, adj_label, labels, weight_tensor_orig, norm, optimizer="Adam", epochs=epochs_T3, lr=lr_T3, save_path=save_path, dataset=dataset)
            epoch_index = epoch_index + 1
        logdict = dict(n_neighbors_comp=n_neighbors_comp, nClusters_2=nClusters_2, acc=acc_best, nmi=nmi_best, ari=ari_best)
        logwriter.writerow(logdict)
        logfile.flush()
