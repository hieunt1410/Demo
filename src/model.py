import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

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
        
        self.init_embed()
        
        self.ub_graph, self.ui_graph, self.bi_graph, self.new_ui_graph = raw_graph
        
        # self.get_aff_graph()
        # self.get_hist_graph()
        # self.get_agg_graph()
        # self.get_aug_bundle_agg_graph()
        
        # self.get_aff_graph_ori()
        # self.get_hist_graph_ori()
        # self.get_agg_graph_ori()
        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        # self.UI_aggregation_graph_ori = self.get_user_prop_graph(self.ui_graph)
        # self.UI_aggregation_graph_ori = self.get_aggregation_graph(self.ui_graph)
        
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)
       
        # self.BI_propagation_graph_ori = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)
        # self.BI_aggregation_graph_ori = self.get_bundle_agg_graph(self.bi_graph)
        
        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, conf['aff_ed_ratio'])
        # self.UI_propagation_graph = self.get_user_prop_graph(self.ui_graph, conf['aff_ed_ratio'])
        # self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, conf['aff_ed_ratio'])
        # self.UI_aug_propagation_graph = self.get_propagation_graph(self.new_ui_graph, conf['aff_ed_ratio'])
        # self.UI_aug_aggregation_graph = self.get_aggregation_graph(self.new_ui_graph, conf['aff_ed_ratio'])
        
        # self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, conf['agg_ed_ratio'])
        
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, conf['agg_ed_ratio'])
        # self.BI_aggregation_graph = self.get_bundle_agg_graph(self.bi_graph, conf['agg_ed_ratio'])
        
        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, conf['hist_ed_ratio'])
        
        # self.fusion_weights = conf['fusion_weights']
        
        self.init_md_dropouts()
        self.init_noise_eps()
        # self.init_fusion_weights()
        # H = mix_graph((self.ub_graph, self.ui_graph, self.bi_graph), self.num_users, self.num_items, self.num_bundles)
        # self.atom_graph = split_hypergraph(normalize_Hyper(H), self.device)
        
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
        self.items_pop = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_pop)
        self.embedding = nn.Embedding(self.num_users + self.num_items, self.embedding_size)
        nn.init.xavier_normal_(self.embedding.weight)
        
    # def init_fusion_weights(self):
    #     assert (len(self.fusion_weights['modal_weight']) == 3), \
    #         "The number of modal fusion weights does not correspond to the number of graphs"

    #     assert (len(self.fusion_weights['UB_layer']) == self.num_layers + 1) and\
    #            (len(self.fusion_weights['UI_layer']) == self.num_layers + 1) and \
    #            (len(self.fusion_weights['BI_layer']) == self.num_layers + 1),\
    #         "The number of layer fusion weights does not correspond to number of layers"
            
    #     modal_coefs = torch.FloatTensor(self.fusion_weights['modal_weight'])
    #     UB_layer_coefs = torch.FloatTensor(self.fusion_weights['UB_layer'])
    #     UI_layer_coefs = torch.FloatTensor(self.fusion_weights['UI_layer'])
    #     BI_layer_coefs = torch.FloatTensor(self.fusion_weights['BI_layer'])

    #     self.modal_coefs = modal_coefs.unsqueeze(-1).unsqueeze(-1).to(self.device)

    #     self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
    #     self.UI_layer_coefs = UI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
    #     self.BI_layer_coefs = BI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        
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
        if modification_ratio:
            graph = birpartite_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            birpartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        
        bundle_sz = birpartite_graph.sum(axis=1) + 1e-8
        # item_freq = birpartite_graph.T.sum(axis=1) + 1e-8
        # birpartite_graph = sp.diags(1/item_freq.A.ravel()) @ birpartite_graph.T
        birpartite_graph = sp.diags(1/bundle_sz.A.ravel()) @ birpartite_graph
        
        return to_tensor(birpartite_graph).to(device)
    
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
        
        degree = np.array(propagation_graph.sum(axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)
        
        # norm_adj = d_mat.dot(propagation_graph).dot(d_mat)
        norm_adj = d_mat @ propagation_graph @ d_mat
        norm_adj = get_sparse_tensor(norm_adj, self.device)
        return norm_adj        
        
        
    def one_propagate(self, graph, Afeat, Bfeat, test):
        device = self.device
        feats = torch.cat((Afeat, Bfeat), dim=0)
        all_feats = [feats]
        
        for i in range(self.num_layers):
            feats = graph @ feats
            feats /= (i + 2)
            
            all_feats.append(F.normalize(feats, p=2, dim=1))
            
        all_feats = torch.stack(all_feats, dim=1)
        all_feats = torch.sum(all_feats, dim=1).squeeze(1)
        
        Afeat, Bfeat = torch.split(all_feats, (Afeat.shape[0], Bfeat.shape[0]), 0)
        
        return Afeat, Bfeat
    
    # def hyper_propagate(self, graph, Ufeat, Ifeat, Bfeat, mess_dropout, test):
    #     feats = torch.cat([Ufeat, Ifeat, Bfeat], dim=0)
    #     all_feats = torch.cat([G @ feats for G in self.atom_graph], dim=0)
    #     all_feats = feats / 2 + mess_dropout(all_feats) / 3
        
    #     Ufeat, Ifeat, Bfeat = torch.split(
    #         all_feats, [Ufeat.shape[0], Ifeat.shape[0], Bfeat.shape[0]], dim=0
    #     )
        
    #     return Ufeat, Ifeat, Bfeat
    
    def one_aggregate(self, agg_graph, node_feature, test):
        aggregated_feature = agg_graph @ node_feature 

        return aggregated_feature
    
    def get_aug_bundle_rep(self, IL_item_feature):
        device = self.device
        bu_graph = self.ub_graph.T
        
        UI_bundle_feature = self.UI_aggregation_graph @ IL_item_feature
        bundle_size = bu_graph.sum(axis=1) + 1e-8
        bu_graph = sp.diags(1/bundle_size.A.ravel()) @ bu_graph
        self.bundle_agg_graph_BU = to_tensor(bu_graph).to(device)
        
        IL_bundle_feature = self.bundle_agg_graph_BU @ UI_bundle_feature
        
        return IL_bundle_feature
    
    def propagate(self, test=False):
        if test:    
            UB_users_feat, UB_bundles_feat = self.one_propagate(self.UB_propagation_graph_ori, self.users_feat, self.bundles_feat, test)
        else:
            UB_users_feat, UB_bundles_feat = self.one_propagate(self.UB_propagation_graph, self.users_feat, self.bundles_feat, test)#user feature in UB view, bundle feature in UB view
            
        if test:
            UI_users_feat, UI_items_feat = self.one_propagate(self.UI_propagation_graph_ori, self.users_feat, self.items_feat - self.items_pop, test)
            
            UI_bundles_feat = self.one_aggregate(self.BI_aggregation_graph_ori, UI_items_feat, test)
            
            # UI_aug_users_feat, UI_aug_items_feat = self.one_propagate(self.UI_aug_propagation_graph, self.users_feat, self.items_feat, 'UI', self.UI_layer_coefs, test)
            # UI_aug_bundles_feat = self.one_aggregate(self.BI_aggregation_graph, UI_aug_items_feat, 'BI', test)
        else:
            UI_users_feat, UI_items_feat = self.one_propagate(self.UI_propagation_graph, self.users_feat, self.items_feat - self.items_pop, test)
            
            UI_bundles_feat = self.one_aggregate(self.BI_aggregation_graph, UI_items_feat, test)#bundle feature in UI view
            
            # UI_aug_users_feat, UI_aug_items_feat = self.one_propagate(self.UI_aug_propagation_graph, self.users_feat, self.items_feat, 'UI', self.UI_layer_coefs, test)
            # UI_aug_bundles_feat = self.one_aggregate(self.BI_aggregation_graph, UI_aug_items_feat, 'BI', test)#bundle feature in UI view    

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
    
    def cal_c_loss(self, pos, aug):
        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)
        
        ttl_score = pos @ aug.T
        ttl_score = torch.sum(torch.exp(ttl_score ), axis=1)
        
        c_loss = -torch.mean(torch.log(torch.exp(pos_score / 1) / ttl_score))
        
        return c_loss
    
    def cal_bpl_loss(self, propagate_result, users, bundles):
        device = self.device
        
        q_list = (self.bi_graph @ (self.ui_graph.sum(axis = 0) + 1e-8).T).A.squeeze()
        
        scores = self.evaluate(propagate_result, users)
        # c_list = groupby_apply(bundles, scores, bins=self.num_bundles, reduction='sum').to(device)
        c_list = [1] * len(q_list)
        
        with np.errstate(invalid='ignore'):
            r_list = c_list/(q_list**(2-1.552))
        
        bpl_loss = self.conf['lambda1']*torch.sqrt(torch.var(r_list))

        return bpl_loss
        
    
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
        
        bundle_c_loss = self.cal_c_loss(aff_bundles_feat, hist_bundles_feat)
        
        aff_users_feat = aff_users_feat[:, 0, :]
        hist_users_feat = hist_users_feat[:, 0, :]
        user_align = self.cal_a_loss(aff_users_feat, hist_users_feat)
        user_uniform = (self.cal_u_loss(aff_users_feat) + self.cal_u_loss(hist_users_feat)) / 2
        user_c_loss = self.cal_c_loss(aff_users_feat, hist_users_feat)
        
        u_loss = (bundle_uniform + user_uniform)
        c_loss = (bundle_c_loss + user_c_loss) 
        a_loss = (bundle_align + user_align)
        
        return bpr_loss, (a_loss + u_loss) / 2

    def forward(self, batch, ED_dropout, psi=1.):
        if ED_dropout:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf['hist_ed_ratio'])
            
            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf['aff_ed_ratio'])
            self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf['aff_ed_ratio'])
            
            # self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf['agg_ed_ratio'])
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf['agg_ed_ratio'])
        
        users, bundles = batch
        users_feat, bundles_feat = self.propagate()
        
        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feat]
        bundles_embedding = [i[bundles] for i in bundles_feat]
        bundles_gamma = torch.tanh(self.bundle_freq / psi)
        bundles_gamma = bundles_gamma[bundles.flatten()].reshape(bundles.shape)
                                                                
        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding, bundles_gamma)
        bpl_loss = self.cal_bpl_loss([users_feat, bundles_feat], users, bundles)
        
        return bpr_loss, c_loss + bpl_loss
        
    def evaluate(self, propagate_result, users, psi=1):
        users_feat, bundles_feat = propagate_result
        aff_users_feat, hist_users_feat = [i[users] for i in users_feat]
        aff_bundles_feat, hist_bundles_feat = bundles_feat
        bundle_gamma = torch.tanh(self.bundle_freq / psi)
        aff_bundles_feat_ =  aff_bundles_feat * (1 - bundle_gamma.unsqueeze(1))
        hist_bundles_feat_ = hist_bundles_feat * bundle_gamma.unsqueeze(1)
        scores = aff_users_feat @ aff_bundles_feat_.T + hist_users_feat @ hist_bundles_feat_.T
        
        return scores