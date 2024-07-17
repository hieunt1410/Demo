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
    # loss = -torch.mean(torch.log(torch.sigmoid(pos - negs)))
    loss = torch.mean(torch.nn.functional.softplus(negs - pos))
    
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

class DenseBatchFCTanh(nn.Module):
    def __init__(self, input_dim, output_dim, reg, do_norm):
        super(DenseBatchFCTanh, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.do_norm = do_norm
        self.batch_norm = nn.BatchNorm1d(output_dim) if do_norm else None
        self.reg = reg

    def forward(self, x):
        x = self.linear(x)
        if self.do_norm:
            x = self.batch_norm(x)
        return torch.tanh(x)

class DenseFC(nn.Module):
    def __init__(self, input_dim, output_dim, reg):
        super(DenseFC, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.reg = reg

    def forward(self, x):
        return self.linear(x)

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
        self.dropout = nn.Dropout(p=0.2 , inplace=True)
        
        self.init_embed()
        
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        
        
        # self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.UI_propagation_graph_ori = self.get_user_prop_graph(self.ui_graph)
        
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)
       
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)
        
        # self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, conf['aff_ed_ratio'])
        self.UI_propagation_graph = self.get_user_prop_graph(self.ui_graph, conf['aff_ed_ratio'])
        
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, conf['agg_ed_ratio'])
        
        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, conf['hist_ed_ratio'])
        
    
    def init_embed(self):
        self.users_feat = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feat)
        self.bundles_feat = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feat)
        self.items_feat = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feat)
        

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
    
    
    def get_user_prop_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])
        
        if modification_ratio:
            graph = propagation_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        
        degree = np.array(propagation_graph.sum(axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)
        
        norm_adj = d_mat @ propagation_graph @ d_mat
        
        return to_tensor(norm_adj).to(device)
        
        
    def one_propagate(self, graph, Afeat, Bfeat, test):
        device = self.device
        feats = torch.cat((Afeat, Bfeat), dim=0)
        ini_feats = F.normalize(feats, p=2, dim=1)
        all_feats = [feats]
        
        for i in range(self.num_layers):
            if isinstance(graph, list):
                feats = graph[i] @ feats
            else:
                feats = graph @ feats

            # feats = self.dropout(feats)
            # feats = feats + self.residual_coff * ini_feats
            # neighbor_feats = self.cal_edge_weight(graph, feats, test)
            # feats = neighbor_feats + self.residual_coff * (feats - ini_feats)
            
            feats /= (i + 2)
            feats = F.normalize(feats, p=2, dim=1)
            
            all_feats.append(feats)
            
        all_feats = torch.stack(all_feats, dim=1)
        all_feats = torch.mean(all_feats, dim=1)
        
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
        mat = 1/2 * torch.exp((2 - 2 * cross_product)/exp_coff) * torch.nn.functional.softplus((2 - 2 * cross_product)/exp_coff)
        # mat = (2 - 2 * cross_product)/exp_coff
        mat = mat * values
        
        new_indices = indices[0].unsqueeze(1).expand(end_emb.shape)
        mat = torch.mul(end_emb, mat.unsqueeze(1).expand(end_emb.shape))
                
        update_all_emb = torch.zeros(emb.shape).to(self.device)
        update_all_emb.scatter_add_(0, new_indices, mat)
        
        return update_all_emb
    
    def one_propagate_(self, graph, Afeat, Bfeat, test):
        device = self.device
        feats = torch.cat((Afeat, Bfeat), dim=0)
        all_feats = [feats]
        
        if not test:
            for i in range(self.num_layers):
                if isinstance(graph, list):
                    feats = graph[i] @ feats
                else:
                    feats = graph @ feats

                feats = self.dropout(feats)
                feats = feats / (i + 2)
                feats = F.normalize(feats, p=2, dim=1)
                
                all_feats.append(feats)
        
        else:
            for i in range(self.num_layers):
                if isinstance(graph, list):
                    feats = graph[i] @ feats
                else:
                    feats = graph @ feats

                feats = self.dropout(feats)
                feats = feats / (i + 2)
                feats = F.normalize(feats, p=2, dim=1)
                
                feats[Afeat.shape[0]:] = Bfeat
                
                all_feats.append(feats)
        
        all_feats = torch.stack(all_feats, dim=1)
        all_feats = torch.mean(all_feats, dim=1)
        
        Afeat, Bfeat = torch.split(all_feats, (Afeat.shape[0], Bfeat.shape[0]), 0)
        
        return Afeat, Bfeat
            
    
    def one_aggregate(self, node_feature, test):
        if test:
            aggregated_feature = self.BI_aggregation_graph_ori @ node_feature
        else:
            aggregated_feature = self.BI_aggregation_graph @ node_feature
        
        return aggregated_feature
    
    def propagate(self, test=False):       
        if test:
            UB_users_feat, UB_bundles_feat = self.one_propagate(self.UB_propagation_graph_ori, self.users_feat, self.bundles_feat, test)
            
        else:
            UB_users_feat, UB_bundles_feat = self.one_propagate(self.UB_propagation_graph, self.users_feat, self.bundles_feat, test)
            
        if test:
            UI_users_feat, UI_items_feat = self.one_propagate(self.UI_propagation_graph_ori, self.users_feat, self.items_feat, test)

        else:
            UI_users_feat, UI_items_feat = self.one_propagate(self.UI_propagation_graph, self.users_feat, self.items_feat, test)
            
                        
        UI_bundles_feat = self.one_aggregate(UI_items_feat, test)
        
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
        pos, neg = bundles[:, 0], bundles[:, 1]
        batch_pop, batch_unpop = split_batch_item(pos, self.bundle_freq)
        
        batch_users = torch.unique(users).type(torch.LongTensor).to(self.device)
        batch_pop = torch.unique(batch_pop).type(torch.LongTensor).to(self.device)
        batch_unpop = torch.unique(batch_unpop).type(torch.LongTensor).to(self.device)
        
        aff_users_feat, hist_users_feat = users_feat
        aff_bundles_feat, hist_bundles_feat = bundles_feat
        
        user_c_loss = InfoNCE(aff_users_feat[batch_users], hist_users_feat[batch_users], 0.2) * 0.2
        
        # bundle_c_pop = InfoNCE_i(aff_users_feat[batch_pop], hist_bundles_feat[batch_pop], hist_bundles_feat[batch_unpop], 0.2, 0.2)
        # bundle_c_unpop = InfoNCE_i(aff_users_feat[batch_unpop], hist_bundles_feat[batch_unpop], hist_bundles_feat[batch_pop], 0.2, 0.2)
        bundle_c_pop = InfoNCE(aff_bundles_feat[batch_pop], hist_bundles_feat[batch_pop], 0.2) * 0.2
        bundle_c_unpop = InfoNCE(aff_bundles_feat[batch_unpop], hist_bundles_feat[batch_unpop], 0.2) * 0.2
        bundle_c_loss = (bundle_c_pop + bundle_c_unpop) * 0.2
        
        c_loss = user_c_loss + bundle_c_loss
        
        return c_loss
    
    # def cal_c_loss(self, users, bundles, users_feat, bundles_feat):
    #     batch_users = users.type(torch.LongTensor).to(self.device)
        
    #     aff_pos_logits = torch.sum(users_feat[0][batch_users] * bundles_feat[0][bundles[:, 0]], 1)
    #     aff_neg_logits = torch.sum(users_feat[0][batch_users] * bundles_feat[0][bundles[:, 1]], 1)
    #     aff_rank_dist = aff_pos_logits - aff_neg_logits
    #     supervised_loss = F.binary_cross_entropy_with_logits(aff_rank_dist, torch.ones_like(aff_rank_dist))
        
    #     hist_pos_logits = torch.sum(users_feat[1][batch_users] * bundles_feat[1][bundles[:, 0]], 1)
    #     hist_neg_logits = torch.sum(users_feat[1][batch_users] * bundles_feat[1][bundles[:, 1]], 1)
    #     hist_rank_dist = hist_pos_logits - hist_neg_logits
        
    #     distill_loss = 0.1 * F.binary_cross_entropy_with_logits(aff_rank_dist, torch.sigmoid(hist_rank_dist))
        
    #     aff_ii_logits = torch.sum(bundles_feat[0][bundles[:, 0]] * bundles_feat[0][bundles[:, 0]], 1)
    #     aff_ij_logits = torch.mean(bundles_feat[0][bundles[:, 0]] @ bundles_feat[0][bundles[:, 1]].T, 1)
    #     aff_iden_dist = aff_ii_logits - aff_ij_logits
        
    #     hist_ii_logits = torch.sum(bundles_feat[1][bundles[:, 0]] * bundles_feat[1][bundles[:, 0]], 1)
    #     hist_ij_logits = torch.mean(bundles_feat[1][bundles[:, 0]] @ bundles_feat[1][bundles[:, 1]].T, 1)
    #     hist_iden_dist = hist_ii_logits - hist_ij_logits
        
    #     distill_loss += F.binary_cross_entropy_with_logits(aff_iden_dist, torch.sigmoid(hist_iden_dist))
    #     distill_loss += 0.1 * (torch.mean(torch.abs(hist_pos_logits - aff_pos_logits)) + torch.mean(torch.abs(hist_neg_logits - aff_neg_logits)))
        
    #     total_loss = supervised_loss + distill_loss
        
    #     return total_loss
    def cal_cl_loss(self, idx, aug_graph_1, aug_graph_2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        b_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        
        u_view_1, b_view_1 = self.one_propagate(aug_graph_1, self.users_feat, self.bundles_feat, True)
        u_view_2, b_view_2 = self.one_propagate(aug_graph_2, self.users_feat, self.bundles_feat, True)
        
        view_1 = torch.cat((u_view_1[u_idx], b_view_1[b_idx]), 0)
        view_2 = torch.cat((u_view_2[u_idx], b_view_2[b_idx]), 0)
        
        return InfoNCE(view_1, view_2, 0.2)
    

    def cal_loss(self, users, bundles, users_feat, bundles_feat, bundles_gamma):
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
        
        aug_graph_1 = self.get_user_prop_graph(self.ub_graph, self.conf['hist_ed_ratio'])
        aug_graph_2 = self.get_user_prop_graph(self.ub_graph, self.conf['hist_ed_ratio'])
        cl_loss = self.cal_cl_loss([users, bundles[:, 0]], aug_graph_1, aug_graph_2)
        
        return bpr_loss, a_loss, u_loss, cl_loss
    
    def l2_reg_loss(self, reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb)
            
        return emb_loss * reg
        

    def forward(self, batch, ED_dropout, psi=1.):
        if ED_dropout:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf['hist_ed_ratio'])
            
            # self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf['aff_ed_ratio'])
            self.UI_propagation_graph = self.get_user_prop_graph(self.ui_graph, self.conf['aff_ed_ratio'])

            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf['agg_ed_ratio'])
        
        users, bundles = batch
        users_feat, bundles_feat = self.propagate()
        
        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feat]
        bundles_embedding = [i[bundles] for i in bundles_feat]
        bundles_gamma = torch.tanh(self.bundle_freq / psi)
        bundles_gamma = bundles_gamma[bundles.flatten()].reshape(bundles.shape)
                                                                
        bpr_loss, a_loss, u_loss, cl_loss = self.cal_loss(users, bundles, users_embedding, bundles_embedding, bundles_gamma)
        c_loss = self.cal_c_loss(users, bundles, users_feat, bundles_feat)
        au_loss = a_loss + u_loss
        
        reg_loss = self.l2_reg_loss(self.l2_norm, self.users_feat[users], self.bundles_feat[bundles[:, 0]], self.bundles_feat[bundles[:, 1]])
        
        return bpr_loss, au_loss, cl_loss + reg_loss / len(users)
        
    def evaluate(self, propagate_result, users, psi=1):
        users_feat, bundles_feat = propagate_result
        aff_users_feat, hist_users_feat = [i[users] for i in users_feat]
        aff_bundles_feat, hist_bundles_feat = bundles_feat
        bundle_gamma = torch.tanh(self.bundle_freq / psi)
        aff_bundles_feat_ =  aff_bundles_feat * (1 - bundle_gamma.unsqueeze(1))
        hist_bundles_feat_ = hist_bundles_feat * bundle_gamma.unsqueeze(1)
        scores = aff_users_feat @ aff_bundles_feat_.T + hist_users_feat @ hist_bundles_feat_.T
        
        return scores