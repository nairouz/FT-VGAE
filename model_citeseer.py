import os
import torch
import csv
import random
import numpy as np
import torch.nn as nn
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from sklearn import metrics
from munkres import Munkres
from torch.nn import Parameter
from sklearn.metrics import f1_score
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering

class ClusterAssignment(nn.Module):
    def __init__(self, cluster_number, embedding_dimension, alpha, cluster_centers=None):
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number, self.embedding_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, inputs):
        norm_squared = torch.sum((inputs.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = random_uniform_init(input_dim, output_dim) 
        self.activation = activation
        
    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs

def generate_comp_index(emb, state, n_neighbors=None, centers=None, y_pred=None):
    if state == 0:
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(emb.detach().cpu().numpy())
        _, indices = nn.kneighbors(emb.detach().cpu().numpy())
    else:
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(emb.detach().cpu().numpy())
        _, indices = nn.kneighbors(centers.detach().cpu().numpy())
        indices = indices[y_pred]
    return indices

def target_distribution(q):
    weight = (q ** 2) / torch.sum(q, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

def computeID(emb, nres=1, fraction=1):
    ID = []
    r = emb.detach().cpu().numpy().astype(np.float64)
    n = int(np.round(r.shape[0] * fraction))            
    dist = squareform(pdist(r, 'euclidean'))
    for i in range(nres):
        dist_s = dist
        perm = np.random.permutation(r.shape[0])[0:n]
        dist_s = dist_s[perm,:]
        dist_s = dist_s[:,perm]
        ID.append(estimate(dist_s)[2]) 
    return ID

def computePC_ID(emb, th=0.9):
    emb_np = emb.detach().cpu().numpy()
    scaler = StandardScaler()
    scaler.fit(emb_np)
    embn = scaler.transform(emb_np)
    pca = PCA()
    pca.fit(embn)
    #cs = np.cumsum(pca.explained_variance_ratio_)
    sv = pca.singular_values_
    evr = pca.explained_variance_ratio_
    cs = np.cumsum(pca.explained_variance_ratio_)
    return np.argwhere(cs > th)[0][0]

def compute_db(sp, feature):
    f_adj = np.matmul(feature, np.transpose(feature))
    predict_labels = sp.fit_predict(feature)
    db = -metrics.davies_bouldin_score(f_adj, predict_labels)
    return db

class FT_VGAE(nn.Module):

    def __init__(self, **kwargs):
        super(FT_VGAE, self).__init__()
        self.n_neighbors_comp = kwargs['n_neighbors_comp']
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters_1 = kwargs['nClusters_1']
        self.nClusters_2 = kwargs['nClusters_2']
        if kwargs['activation'] == "ReLU":
            self.activation = F.relu
        if kwargs['activation'] == "Sigmoid":
            self.activation = F.sigmoid
        if kwargs['activation'] == "Tanh":
            self.activation = F.tanh
        self.alpha = kwargs['alpha']
        self.gamma_1 = kwargs['gamma_1']
        self.gamma_2 = kwargs['gamma_2']

        # VGAE training parameters
        self.base_gcn = GraphConvSparse(self.num_features, self.num_neurons, self.activation)
        self.gcn_mean = GraphConvSparse(self.num_neurons, self.embedding_size, activation=lambda x:x)
        self.gcn_logsigma2 = GraphConvSparse(self.num_neurons, self.embedding_size, activation=lambda x:x)
        self.assignment_1 = ClusterAssignment(self.nClusters_1, self.embedding_size, self.alpha)
        self.kl_loss = nn.KLDivLoss(size_average=False) 
                    
    def train_1(self, epoch_index, adj_norm, features, adj_label, y, weight_tensor, norm, optimizer="Adam", epochs=200, lr=0.01, save_path="/home/mrabah_n/code/ICDM/Clus_VGAE/results/", dataset="Cora"):
        if optimizer ==  "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay=0.01)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay=0.01)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        epoch_bar=tqdm(range(epochs))

        ##############################################################
        # Training loop
        print('Training......')
        acc_best = 0
        FT_best = 0
        epoch_best = epoch_index
        for epoch in epoch_bar:
            opti.zero_grad()
            z_mu, z_sigma2_log, emb, hidden = self.encode(features, adj_norm) 
            q_1 = self.assignment_1(z_mu)
            adj_out = self.decode(emb)

            ##############################################################
            # Making 3D tensors for compactness loss
            indices_comp = generate_comp_index(emb, state=0, n_neighbors=self.n_neighbors_comp)
            z_mu_comp = z_mu[indices_comp,:]
            z_sigma2_log_comp = z_sigma2_log[indices_comp,:]
            adj_out_comp = adj_out[indices_comp,:]
            adj_out_orig_com = adj_label.to_dense().unsqueeze(1).repeat_interleave(self.n_neighbors_comp, dim=1)
            weight_tensor_orig_com = weight_tensor.view(adj_label.shape).unsqueeze(1).repeat_interleave(self.n_neighbors_comp, dim=1)
          
            ##############################################################
            # Loss
            Loss_recons = adj_out_orig_com * torch.log(adj_out_comp + 1e-10) + (1 - adj_out_orig_com) * torch.log(1- adj_out_comp + 1e-10)
            Loss_recons = - norm * torch.mean(torch.sum(Loss_recons * weight_tensor_orig_com, dim=2))
            Loss_reg = torch.mean(torch.sum((-1 / adj_out.size(0)) * (1 + z_sigma2_log_comp - z_mu_comp**2 - torch.exp(z_sigma2_log_comp)), dim=2))
            Loss_elbo = Loss_recons + Loss_reg 

            ##############################################################
            # Evaluation
            y_pred = self.predict(q_1)
            ID = []
            PC_ID = []
            for k in range(self.nClusters_1):
                ID.append(computeID(emb[y_pred==k]))
                PC_ID.append(computePC_ID(emb[y_pred==k]))
            ID = np.asarray(ID)
            PC_ID = np.asarray(PC_ID)
            ID_global_mean = np.mean(ID)
            ID_local_mean = np.mean(ID, axis=1) 
            ID_global_error = np.std(ID)
            ID_local_error = np.std(ID, axis=1)
            PC_ID_mean = np.mean(PC_ID)
            FT = PC_ID_mean - ID_global_mean

            ##############################################################
            # Update learnable parameters
            Loss_elbo.backward()
            opti.step()
            lr_s.step()

            ##############################################################
            #Save logs  
            if FT > FT_best:
                FT_best = FT
                epoch_best = epoch + epoch_index
                torch.save(self.state_dict(), save_path + dataset + '/T1/model_best.pk')
        epoch_index = epoch_best
        return epoch_index
          
    def train_2(self, acc_best, nmi_best, ari_best, epoch_index, adj_norm, features, adj_label, y, weight_tensor, norm, optimizer="Adam", epochs=200, lr=0.01, save_path="/home/mrabah_n/code/ICDM/Clus_VGAE/results/", dataset="Cora"):
        assignment_2 = ClusterAssignment(self.nClusters_2, self.embedding_size, self.alpha).to("cuda:0")
        if optimizer ==  "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay=0.001)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay=0.01)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        epoch_bar = tqdm(range(epochs))
        km = KMeans(n_clusters=self.nClusters_2, n_init=20)
        sp = SpectralClustering(n_clusters=self.nClusters_1)
        with torch.no_grad():
            z_mu, z_sigma2_log, emb, _ = self.encode(features, adj_norm)
            km.fit(z_mu.detach().cpu().numpy())
            centers_2 = torch.tensor(km.cluster_centers_, dtype=torch.float, requires_grad=True) 
            assignment_2.state_dict()["cluster_centers"].copy_(centers_2)
        
        ##############################################################
        # Training loop
        for epoch in epoch_bar:
            opti.zero_grad()
            z_mu, z_sigma2_log, emb, hidden = self.encode(features, adj_norm)
            q_1 = self.assignment_1(z_mu) 
            q_2 = assignment_2(z_mu)
            p_2 = target_distribution(q_2.detach().cpu()).to("cuda:0")
            adj_out = self.decode(emb)

            ##############################################################
            # Loss
            Loss_recons = norm * F.binary_cross_entropy(adj_out.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
            Loss_reg = - 1 / adj_out.size(0) * (1 + z_sigma2_log - z_mu**2 - torch.exp(z_sigma2_log)).sum(1).mean()
            Loss_clus = self.kl_loss(torch.log(q_2), p_2)
            Loss_elbo = Loss_recons + Loss_clus + Loss_reg
            
            ##############################################################
            # Evaluation                        
            y_pred_spec = sp.fit_predict(emb.detach().cpu().numpy())
            cm = clustering_metrics(y, y_pred_spec)
            acc_spec, nmi_spec, ari_spec, _, _, _, _ = cm.evaluationClusterModelFromLabel()
            if acc_best < acc_spec:
                acc_best, nmi_best, ari_best = acc_spec, nmi_spec, ari_spec

            ##############################################################
            # Update learnable parameters
            Loss_elbo.backward()
            opti.step()
            lr_s.step()

        epoch_index = epoch + epoch_index
        return acc_best, nmi_best, ari_best, epoch_index

    def train_3(self, acc_best, nmi_best, ari_best, epoch_index, adj_norm, features, adj_label, y, weight_tensor, norm, optimizer="Adam", epochs=200, lr=0.01, save_path="/home/mrabah_n/code/ICDM/Clus_VGAE/results/", dataset="Cora"):
        if optimizer ==  "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay=0.001)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay=0.01)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        epoch_bar=tqdm(range(epochs))
        sp = SpectralClustering(n_clusters=self.nClusters_1)

        ##############################################################
        # Training loop
        print('Training......')
        for epoch in epoch_bar:
            opti.zero_grad()
            z_mu, z_sigma2_log, emb, hidden = self.encode(features, adj_norm) 
            q_1 = self.assignment_1(z_mu)
            p_1 = target_distribution(q_1.detach().cpu()).to("cuda:0")
            adj_out = self.decode(emb)

            # Making 3D tensors for compactness loss
            indices_comp = generate_comp_index(emb, state=1, centers=self.assignment_1.state_dict()["cluster_centers"], y_pred=self.predict(q_1))
            z_mu_comp = z_mu[indices_comp,:]
            z_sigma2_log_comp = z_sigma2_log[indices_comp,:]
            adj_out_comp = adj_out[indices_comp,:]
            adj_out_orig_com = adj_label.to_dense().unsqueeze(1).repeat_interleave(1, dim=1)
            weight_tensor_orig_com = weight_tensor.view(adj_label.shape).unsqueeze(1).repeat_interleave(1, dim=1)
            
            ##############################################################
            # Loss
            Loss_recons = adj_out_orig_com * torch.log(adj_out_comp + 1e-10) + (1 - adj_out_orig_com) * torch.log(1- adj_out_comp + 1e-10)
            Loss_recons = - norm * torch.mean(torch.sum(Loss_recons * weight_tensor_orig_com, dim=2))
            Loss_reg =  torch.mean(torch.sum((-1 / adj_out.size(0)) * (1 + z_sigma2_log_comp - z_mu_comp**2 - torch.exp(z_sigma2_log_comp)), dim=2))
            Loss_clus = self.kl_loss(torch.log(q_1), p_1)
            Loss_elbo = Loss_recons + Loss_clus + Loss_reg

            ##############################################################
            # Evaluation
            y_pred_spec = sp.fit_predict(emb.detach().cpu().numpy())
            cm = clustering_metrics(y, y_pred_spec)
            acc_spec, nmi_spec, ari_spec, _, _, _, _ = cm.evaluationClusterModelFromLabel()
            if acc_best < acc_spec:
                acc_best, nmi_best, ari_best = acc_spec, nmi_spec, ari_spec

            ##############################################################
            # Update learnable parameters
            Loss_elbo.backward()
            opti.step()
            lr_s.step()

        epoch_index = epoch + epoch_index
        return acc_best, nmi_best, ari_best, epoch_index

    def predict(self, q):
        #
        return np.argmax(q.detach().cpu().numpy(), axis=1)

    def encode(self, x_features, adj):
        hidden = self.base_gcn(x_features, adj)
        mean = self.gcn_mean(hidden, adj)
        log_sigma2 = self.gcn_logsigma2(hidden, adj)
        gaussian_noise = torch.randn(x_features.size(0), self.embedding_size).to("cuda:0")
        #noise = torch.FloatTensor(x_features.size(0), self.embedding_size).uniform_(0, 0.5)
        sampled_z = gaussian_noise * torch.exp(log_sigma2 / 2) + mean
        #sampled_z = self.mean
        return mean, log_sigma2, sampled_z, hidden
            
    @staticmethod
    def decode(z):
        A_pred = torch.sigmoid(torch.matmul(z,z.t()))
        return A_pred
                 
def random_uniform_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)
  
class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

        fh = open('recoder.txt', 'a')

        fh.write('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore) )
        fh.write('\r\n')
        fh.flush()
        fh.close()

        return acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro
