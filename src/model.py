import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
import pickle

def cal_bpr_loss(pred):
    negs =  pred[:, 1].unsqueeze(1)
    pos = pred[:, 0].unsqueeze(1)
    loss = -torch.mean(torch.log(torch.sigmoid(pos - negs)))
    
    return loss

def laplace_transform(graph):
    epsilon = 1e-8
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + epsilon))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + epsilon))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    
    return graph

def np_edge_dropout(values, dropout):
    mask = np.random.choice([0, 1], size = len(values), p=[dropout, 1 - dropout])
    values = mask * values
    
    return values

class Demo(nn.Module):
    def __init__(self, conf, raw_graph, bundles_freq):
        super(Demo, self).__init__()
        self.conf = conf
        self.device = conf['device']
        self.embedding_size = conf['embedding_size']
        self.l2_norm = conf['lambda2']
        self.num_users = conf['num_users']
        self.num_bundles = conf['num_bundles']
        self.num_items = conf['num_items']
        self.num_layers = conf['num_layers']
        self.bundle_freq = torch.FloatTensor(bundles_freq).to(self.device)
        
        self.residual_coff = conf['residual_coff']
        
        self.init_embed()
        
        self.ub_graph, self.ui_graph, self.bi_graph, self.new_ui_graph = raw_graph
        
        
        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        # self.UI_propagation_graph_ori = self.get_user_prop_graph(self.ui_graph)
        
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)
       
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)
        # self.BI_aggregation_graph_ori = self.get_bundle_agg_graph(self.bi_graph)
        
        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, conf['aff_ed_ratio'])
        # self.UI_propagation_graph = self.get_user_prop_graph(self.ui_graph, conf['aff_ed_ratio'])
        
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, conf['agg_ed_ratio'])
        # self.BI_aggregation_graph = self.get_bundle_agg_graph(self.bi_graph, conf['agg_ed_ratio'])
        
        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, conf['hist_ed_ratio'])
        
        self.init_md_dropouts()
        self.init_noise_eps()
        
        
        
    def init_md_dropouts(self):
        self.UB_dropout = nn.Dropout(self.conf['hist_ed_ratio'])
        self.UI_dropout = nn.Dropout(self.conf['aff_ed_ratio'])
        self.BI_dropout = nn.Dropout(self.conf['agg_ed_ratio'])
        self.mess_dropout_dict = {
            'UB': self.UB_dropout,
            'UI': self.UI_dropout,
            'BI': self.BI_dropout
        }
        
    def init_noise_eps(self):
        self.UB_eps = self.conf["hist_ed_ratio"]
        self.UI_eps = self.conf["aff_ed_ratio"]
        self.BI_eps = self.conf["agg_ed_ratio"]
        self.eps_dict = {
            "UB": self.UB_eps,
            "UI": self.UI_eps,
            "BI": self.BI_eps
        }
    
    def init_embed(self):
        self.users_feat = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feat)
        self.bundles_feat = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feat)
        self.items_feat = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feat)
        self.A = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.A)

    def get_propagation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])
        
        if modification_ratio:
            graph = propagation_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        
        return to_tensor(laplace_transform(propagation_graph)).to(device)
    
    def get_aggregation_graph(self, birpartite_graph, modification_ratio=0):
        device = self.device
        # graph = birpartite_graph.tocoo()
        # be = []
        # for b in range(birpartite_graph.shape[0]):
        #     idx = birpartite_graph[b].nonzero()[1]
        #     w = F.softmax(torch.Tensor(self.ui_graph.T[idx].sum(axis=1).tolist()), 0).to(device)
        #     be += w.reshape(1, -1).tolist()[0]

        # birpartite_graph = sp.coo_matrix((be, (graph.row, graph.col)), shape=graph.shape).tocsr()
        with open(self.conf['data_path'] + self.conf['dataset'] + 'bun_atten_graph.pkl', 'rb') as f:
            birpartite_graph = pickle.load(f)
            
        if modification_ratio:
            graph = birpartite_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            birpartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        
        return to_tensor(birpartite_graph).to(device)
    
    # def get_aggregation_graph(self, birpartite_graph, modification_ratio=0):
    #     device = self.device

    #     if modification_ratio:
    #         graph = birpartite_graph.tocoo()
    #         values = np_edge_dropout(graph.data, modification_ratio)
    #         birpartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        
    #     bundle_sz = birpartite_graph.sum(axis=1) + 1e-8
    #     birpartite_graph = sp.diags(1/bundle_sz.A.ravel()) @ birpartite_graph
        
    #     return to_tensor(birpartite_graph).to(device)
    
    def get_bundle_agg_graph(self, birpartite_graph, modification_ratio=0):
        device = self.device
        if modification_ratio:
            graph = birpartite_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            birpartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
            
        items_pop = self.ui_graph.T @ self.ui_graph
        
        return to_tensor(birpartite_graph @ items_pop).to(device)
    
    def get_user_prop_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])
        
        if modification_ratio:
            graph = propagation_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        
        degree = np.array(propagation_graph.sum(axis=0)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)
        
        # norm_adj = d_mat.dot(propagation_graph).dot(d_mat)
        norm_adj = d_mat @ propagation_graph @ d_mat
        
        return to_tensor(norm_adj).to(device)        
        
        
    def one_propagate(self, graph, Afeat, Bfeat, test):
        device = self.device
        feats = torch.cat((Afeat, Bfeat), dim=0)
        ini_feats = F.normalize(feats, p=2, dim=1)
        all_feats = [feats]
        
        for i in range(self.num_layers):
            feats = graph @ feats
            # feats /= (i + 2)
            if not test:
                sign = torch.sign(feats)
                random_noise = F.normalize(torch.rand(feats.shape).to(self.device)) * 0.1
                feats = feats + sign * random_noise
            feats = feats + self.residual_coff * ini_feats   
            # neighbor_feats = self.cal_edge_weight(graph, feats, test)
            # feats = neighbor_feats + self.residual_coff * (feats - ini_feats)
            feats /= (i + 2)
            feats = F.normalize(feats, p=2, dim=1)
            
            all_feats.append(feats)
            
        all_feats = torch.stack(all_feats, dim=1)
        # all_feats = torch.mean(all_feats, dim=1)
        all_feats = torch.sum(all_feats, dim=1)
        
        Afeat, Bfeat = torch.split(all_feats, (Afeat.shape[0], Bfeat.shape[0]), 0)
        
        return Afeat, Bfeat
    
    def cal_edge_weight(self, prop_graph, emb, test):
        prop_graph = prop_graph.coalesce()
        indices = prop_graph._indices().to(self.device)
        values = prop_graph._values().to(self.device)
        
        if not test:
            sign = torch.sign(emb)
            random_noise = F.normalize(torch.rand(emb.shape).to(self.device)) * 0.1
            emb = emb + sign * random_noise
        
        start_emb = emb[indices[0]]
        end_emb = emb[indices[1]]
        cross_product = torch.mul(start_emb, end_emb).mean(dim=1)
        
        exp_coff = 0.5
        # mat = 1/2 * torch.exp((2 - 2 * cross_product)/exp_coff) * torch.nn.functional.softplus((2 - 2 * cross_product)/exp_coff)
        mat = (2 - 2 * cross_product)/exp_coff
        mat = mat * values
        
        new_indices = indices[0].unsqueeze(1).expand(end_emb.shape)
        mat = torch.mul(self.mess_dropout_dict['UI'](end_emb), mat.unsqueeze(1).expand(end_emb.shape))
                
        update_all_emb = torch.zeros(emb.shape).to(self.device)
        update_all_emb.scatter_add_(0, new_indices, mat)
        
        return update_all_emb
    
    def one_aggregate(self, bundle_agg_graph, node_feature, test):
        aggregated_feature = bundle_agg_graph @ node_feature
        
        return aggregated_feature
    
    def propagate(self, test=False):       
        if test:
            UB_users_feat, UB_bundles_feat = self.one_propagate(self.UB_propagation_graph_ori, self.users_feat, self.bundles_feat, test)
        else:
            UB_users_feat, UB_bundles_feat = self.one_propagate(self.UB_propagation_graph, self.users_feat, self.bundles_feat, test)#user feature in UB view, bundle feature in UB view
            
        if test:
            UI_users_feat, UI_items_feat = self.one_propagate(self.UI_propagation_graph_ori, self.users_feat, self.items_feat, test)
            
            UI_bundles_feat = self.one_aggregate(self.BI_aggregation_graph_ori, UI_items_feat, test)

        else:
            UI_users_feat, UI_items_feat = self.one_propagate(self.UI_propagation_graph, self.users_feat, self.items_feat, test)
            
            UI_bundles_feat = self.one_aggregate(self.BI_aggregation_graph, UI_items_feat, test)#bundle feature in UI view

        # IL_bundle_feature = self.get_aug_bundle_rep(UI_items_feat)#UI_bundle_feature
        
        aff_users_rep, aff_bundles_rep = UI_users_feat, UI_bundles_feat
        hist_users_rep, hist_bundles_rep = UB_users_feat, UB_bundles_feat
        
        return [aff_users_rep, hist_users_rep], [aff_bundles_rep, hist_bundles_rep]
            
    def cal_a_loss(self, x, y):
        x, y = F.normalize(x, p=2, dim=1), F.normalize(y, p=2, dim=1)       
        return (x - y).norm(p=2, dim=1).pow(2).mean()
    
    def cal_u_loss(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()
    
    def cal_c_loss(self, users, bundles, users_feat, bundles_feat):
        # pos = F.normalize(pos, p=2, dim=1)
        # aug = F.normalize(aug, p=2, dim=1)
        # pos_score = torch.sum(pos * aug, dim=1)
        
        # ttl_score = pos @ aug.T
        # ttl_score = torch.sum(torch.exp(ttl_score ), axis=1)
        
        # c_loss = -torch.mean(torch.log(torch.exp(pos_score / 1) / ttl_score))
        
        # return c_loss
        pos, neg = bundles[:, 0], bundles[:, 1]
        batch_pop, batch_unpop = split_batch_item(pos, self.bundle_freq)
        
        batch_users = torch.unique(users).type(torch.LongTensor).to(self.device)
        batch_pop = torch.unique(batch_pop).type(torch.LongTensor).to(self.device)
        batch_unpop = torch.unique(batch_unpop).type(torch.LongTensor).to(self.device)
        
        aff_users_feat, hist_users_feat = users_feat
        aff_bundles_feat, hist_bundles_feat = bundles_feat
        
        user_c_loss = InfoNCE(aff_users_feat[batch_users], hist_users_feat[batch_users], 0.2)
        bundle_c_pop = InfoNCE_i(aff_users_feat[batch_pop], hist_bundles_feat[batch_pop], hist_bundles_feat[batch_unpop], 0.2, 0.2)
        bundle_c_unpop = InfoNCE_i(aff_users_feat[batch_unpop], hist_bundles_feat[batch_unpop], hist_bundles_feat[batch_pop], 0.2, 0.2)
        
        bundle_c_loss = (bundle_c_pop + bundle_c_unpop) * 0.5
        c_loss = user_c_loss + bundle_c_loss
        
        return c_loss
        

    def cal_loss(self, users_feat, bundles_feat, bundles_gamma):
        aff_users_feat, hist_users_feat = users_feat
        aff_bundles_feat, hist_bundles_feat = bundles_feat
        aff_bundles_feat_ = aff_bundles_feat * (1 - bundles_gamma.unsqueeze(2))
        hist_bundles_feat_ = hist_bundles_feat * bundles_gamma.unsqueeze(2)
        
        pred = torch.sum(aff_users_feat * aff_bundles_feat_, 2) + torch.sum(hist_users_feat * hist_bundles_feat_, 2)
        bpr_loss = cal_bpr_loss(pred)
        
        aff_bundles_feat = aff_bundles_feat[:, 0, :]
        hist_bundles_feat = hist_bundles_feat[:, 0, :]
        bundle_align = self.cal_a_loss(aff_bundles_feat, hist_bundles_feat)
        bundle_uniform = (self.cal_u_loss(aff_bundles_feat) + self.cal_u_loss(hist_bundles_feat)) / 2
        
        aff_users_feat = aff_users_feat[:, 0, :]
        hist_users_feat = hist_users_feat[:, 0, :]
        user_align = self.cal_a_loss(aff_users_feat, hist_users_feat)
        user_uniform = (self.cal_u_loss(aff_users_feat) + self.cal_u_loss(hist_users_feat)) / 2
        
        u_loss = (bundle_uniform + user_uniform)
        a_loss = (bundle_align + user_align)
        
        return bpr_loss, a_loss, u_loss

    def forward(self, batch, ED_dropout, psi=1.):
        if ED_dropout:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf['hist_ed_ratio'])
            
            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf['aff_ed_ratio'])

            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf['agg_ed_ratio'])
        
        users, bundles = batch
        users_feat, bundles_feat = self.propagate()
        
        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feat]
        bundles_embedding = [i[bundles] for i in bundles_feat]
        bundles_gamma = torch.tanh(self.bundle_freq / psi)
        bundles_gamma = bundles_gamma[bundles.flatten()].reshape(bundles.shape)
                                                                
        bpr_loss, a_loss, u_loss = self.cal_loss(users_embedding, bundles_embedding, bundles_gamma)
        c_loss = self.cal_c_loss(users, bundles, users_feat, bundles_feat)
        c_loss = (a_loss + u_loss) / 2
        
        return bpr_loss, c_loss
        
    def evaluate(self, propagate_result, users, psi=1):
        users_feat, bundles_feat = propagate_result
        aff_users_feat, hist_users_feat = [i[users] for i in users_feat]
        aff_bundles_feat, hist_bundles_feat = bundles_feat
        bundle_gamma = torch.tanh(self.bundle_freq / psi)
        aff_bundles_feat_ =  aff_bundles_feat * (1 - bundle_gamma.unsqueeze(1))
        hist_bundles_feat_ = hist_bundles_feat * bundle_gamma.unsqueeze(1)
        scores = aff_users_feat @ aff_bundles_feat_.T + hist_users_feat @ hist_bundles_feat_.T
        
        return scores