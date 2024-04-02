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
        self.UI_aggregation_graph_ori = self.get_aggregation_graph(self.ui_graph)
        
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)
       
        self.BI_propagation_graph_ori = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)
        
        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, conf['aff_ed_ratio'])
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, conf['aff_ed_ratio'])
        
        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, conf['agg_ed_ratio'])
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, conf['agg_ed_ratio'])
        
        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, conf['hist_ed_ratio'])
        
        self.init_md_dropouts()
        
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
    
    def init_embed(self):
        self.users_feat = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feat)
        self.bundles_feat = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feat)
        self.items_feat = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feat)
        self.IL_layer = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        nn.init.xavier_normal_(self.IL_layer.weight)
        self.BL_layer = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        nn.init.xavier_normal_(self.BL_layer.weight)
        
    def init_fusion_weights(self):
        modal_coefs = torch.FloatTensor([0.5, 0.2, 0.3])
        UB_layer_coefs = torch.FloatTensor([0.35, 0.15, 0.5])
        UI_layer_coefs = torch.FloatTensor([0.25, 0.65, 0.1])
        BI_layer_coefs = torch.FloatTensor([0.4, 0.4, 0.2])

        self.modal_coefs = modal_coefs.unsqueeze(-1).unsqueeze(-1).to(self.device)

        self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.UI_layer_coefs = UI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.BI_layer_coefs = BI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        
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
        birpartite_graph = sp.diags(1/bundle_sz.A.ravel()) @ birpartite_graph
        
        return to_tensor(birpartite_graph).to(device)
    
    def one_propagate(self, graph, Afeat, Bfeat, graph_type, test):
        mess_dropout = self.mess_dropout_dict[graph_type]
        
        feats = torch.cat((Afeat, Bfeat), dim=0)
        all_feats = [feats]
        
        for i in range(self.num_layers):
            feats = torch.spmm(graph, feats)
            if not test:
                feat = mess_dropout(feats)
            feats = feats / (i+2)
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
    
    def one_aggregate(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = agg_graph @ node_feature
        
        mess_dropout = self.mess_dropout_dict[graph_type]

        return aggregated_feature if not test else mess_dropout(aggregated_feature)
    
    def fuse(self, users_feature, bundles_feature):
        users_feature = torch.stack(users_feature, dim=0)
        bundles_feature = torch.stack(bundles_feature, dim=0)
        
        users_rep = torch.sum(users_feature * self.modal_coefs, dim=0)
        bundles_rep = torch.sum(bundles_feature * self.modal_coefs, dim=0)
        
        return users_rep, bundles_rep
    
    def propagate(self, test=False):
        if test:
            UB_users_feat, UB_bundles_feat = self.one_propagate(self.UB_propagation_graph_ori, self.users_feat, self.bundles_feat, 'UB', test)
        else:
            UB_users_feat, UB_bundles_feat = self.one_propagate(self.UB_propagation_graph, self.users_feat, self.bundles_feat, 'UB', test)
            
        if test:
            UI_users_feat, UI_items_feat = self.one_propagate(self.UI_propagation_graph_ori, self.users_feat, self.items_feat, 'UI', test)
            UI_bundles_feat = self.one_aggregate(self.UI_aggregation_graph_ori, self.items_feat, 'BI', test)
        else:
            UI_users_feat, UI_items_feat = self.one_propagate(self.UI_propagation_graph, self.users_feat, self.items_feat, 'UI', test)
            UI_bundles_feat = self.one_aggregate(self.UI_aggregation_graph, self.items_feat, 'BI', test)
            
        if test:
            BI_bundles_feat, BI_items_feat = self.one_propagate(self.BI_propagation_graph_ori, self.bundles_feat, self.items_feat, 'BI', test)
            BI_users_feat = self.one_aggregate(self.UI_aggregation_graph_ori, BI_items_feat, 'UI', test)
        else:
            BI_bundles_feat, BI_items_feat = self.one_propagate(self.BI_propagation_graph, self.bundles_feat, self.items_feat, 'BI', test)
            BI_users_feat = self.one_aggregate(self.UI_aggregation_graph, BI_items_feat, 'UI', test)            
            
        users_feature = [UB_users_feat, UI_users_feat, BI_users_feat]
        bundles_feature = [UB_bundles_feat, UI_bundles_feat, BI_bundles_feat]
        
        aff_users_rep, aff_bundles_rep = (UI_users_feat+UB_users_feat@BI_users_feat)/2, (UI_bundles_feat+UB_bundles_feat@BI_bundles_feat)/2
        hist_users_rep, hist_bundles_rep = UB_users_feat, UB_bundles_feat
        
        return [hist_users_rep, aff_users_rep], [hist_bundles_rep, aff_bundles_rep]
            
    def cal_a_loss(self, x, y):
        x, y = F.normalize(x, p=2, dim=1), F.normalize(y, p=2, dim=1)       
        return (x - y).norm(p=2, dim=1).pow(2).mean()
    
    def cal_u_loss(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()
    
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
        bundle_c_loss = bundle_align + bundle_uniform
        
        aff_users_feat = aff_users_feat[:, 0, :]
        hist_users_feat = hist_users_feat[:, 0, :]
        user_align = self.cal_a_loss(aff_users_feat, hist_users_feat)
        user_uniform = (self.cal_u_loss(aff_users_feat) + self.cal_u_loss(hist_users_feat)) / 2
        user_c_loss = user_align + user_uniform
        
        c_loss = (bundle_c_loss + user_c_loss) / 2
        
        return bpr_loss, c_loss
    
    def forward(self, batch, ED_dropout, psi=1.):
        if ED_dropout:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf['hist_ed_ratio'])
            
            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf['aff_ed_ratio'])
            self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf['aff_ed_ratio'])
            
            self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf['agg_ed_ratio'])
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf['agg_ed_ratio'])
        
        users, bundles = batch
        users_feat, bundles_feat = self.propagate()
        
        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feat]
        bundles_embedding = [i[bundles] for i in bundles_feat]
        bundles_gamma = torch.tanh(self.bundle_freq / psi)
        bundles_gamma = bundles_gamma[bundles.flatten()].reshape(bundles.shape)
                                                                
        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding, bundles_gamma)
        
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