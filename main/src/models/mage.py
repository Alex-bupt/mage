import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_knn_normalized_graph
from utils.paths import dataset_path
from collections import defaultdict
import json
import numpy as np
import scipy.sparse as sp
import os
import sys


# ============================================================
# [Debug] Numeric anomaly checker
# ============================================================
def check_tensor(name, tensor, stop_on_error=True):
    if tensor is None:
        return False
    if torch.isnan(tensor).any():
        print(f"\n[CRITICAL ERROR] Found NaN in: {name}")
        if stop_on_error: sys.exit(1)
        return True
    if torch.isinf(tensor).any():
        print(f"\n[CRITICAL ERROR] Found Inf in: {name}")
        if stop_on_error: sys.exit(1)
        return True
    return False


# ============================================================
# BiLinearGate
# ============================================================
class BiLinearGate(nn.Module):
    def __init__(self, dim, reduction_ratio=8):
        super(BiLinearGate, self).__init__()
        hidden_dim = max(dim // reduction_ratio, 16)
        self.gate_v_maker = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
        self.v_self_calib = nn.Linear(dim, dim, bias=False)
        self.gate_t_maker = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
        self.t_self_calib = nn.Linear(dim, dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.residual_alpha = 0.5

    def forward(self, x_v, x_t):
        suggestion_from_text = self.gate_t_maker(x_t)
        defense_from_visual = self.v_self_calib(x_v)
        g_v = self.sigmoid(suggestion_from_text + defense_from_visual)

        suggestion_from_visual = self.gate_v_maker(x_v)
        defense_from_text = self.t_self_calib(x_t)
        g_t = self.sigmoid(suggestion_from_visual + defense_from_text)

        x_v_gated = x_v + self.residual_alpha * (x_v * g_v)
        x_t_gated = x_t + self.residual_alpha * (x_t * g_t)
        return x_v_gated, x_t_gated


# ============================================================
# Model (Asymmetric Dual-Stream Implementation)
# ============================================================
class MAGE(GeneralRecommender):
    def __init__(self, config, dataset, local_time):
        super(MAGE, self).__init__(config, dataset)

        self.sparse = True
        self.bm_loss = config['bm_loss']
        self.um_loss = config['um_loss']
        self.vt_loss = config['vt_loss']
        self.reg_weight_1 = config['reg_weight_1']
        self.reg_weight_2 = config['reg_weight_2']
        self.bm_temp = config['bm_temp']
        self.um_temp = config['um_temp']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.device = config['device']

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.extended_iGUME_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_iGUME_user.weight)
        self.extended_text_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_text_user.weight)

        self.dataset_path = dataset_path(config)
        self.p2b_weight = float(config['p2b_weight'])
        max_p2b_k = config['max_p2b_k'] if 'max_p2b_k' in config else 20
        if isinstance(max_p2b_k, bool) or not isinstance(max_p2b_k, (int, np.integer)) or max_p2b_k < 0:
            raise ValueError('max_p2b_k must be a non-negative integer')
        self.max_p2b_k = int(max_p2b_k)
        self.last_loss_components = {}

        # Independent graph ablations. Defaults retain the deployed model's
        # operators and parameter initialization; neither option changes data.
        self.visual_user_graph = config['visual_user_graph'] if 'visual_user_graph' in config else 'shared_text'
        self.explicit_visual_graph = config['explicit_visual_graph'] if 'explicit_visual_graph' in config else 'raw_knn'
        if self.visual_user_graph not in ('shared_text', 'own_visual'):
            raise ValueError('visual_user_graph must be shared_text or own_visual')
        if self.explicit_visual_graph not in ('raw_knn', 'consensus'):
            raise ValueError('explicit_visual_graph must be raw_knn or consensus')

        self.bilinear_fusion = BiLinearGate(self.embedding_dim)

        # Load P2B map (optional)
        self.p2b_map = {}
        p2b_file = os.path.join(self.dataset_path, 'p2b_map_filtered.json')
        if os.path.exists(p2b_file):
            with open(p2b_file, 'r', encoding='utf-8') as f:
                self.p2b_map = json.load(f)

        iGUME_adj_file = os.path.join(self.dataset_path, 'iGUME_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(self.dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        # 1) Load raw features and build KNN graphs
        if self.v_feat is not None:
            self.iGUME_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(iGUME_adj_file):
                iGUME_adj = torch.load(iGUME_adj_file, map_location='cpu', weights_only=True)
            else:
                iGUME_adj = build_sim(self.iGUME_embedding.weight.detach())
                iGUME_adj = build_knn_normalized_graph(iGUME_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(iGUME_adj, iGUME_adj_file)
            self.iGUME_original_adj = iGUME_adj.to(self.device)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file, map_location='cpu', weights_only=True)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.to(self.device)

        # 2) Build shared graph connections
        self.inter = self.find_inter(self.iGUME_original_adj, self.text_original_adj)

        # Build two item-item graphs
        # 2.1 Visual-only graph: KNN similarity only (no P2B)
        print("[Model] Building visual graph (KNN only)...")
        self.ii_adj_visual = self.add_edge(self.inter, p2b_map=None)

        # The paper's visual item stream uses visual/text KNN consensus.
        # Test only the explicit item operator: extended UI, text, and ID
        # operators below remain unchanged. Do not add loops or reverse edges.
        self.explicit_visual_adj = self.iGUME_original_adj
        if self.explicit_visual_graph == 'consensus':
            consensus_adj = self.normalize_item_adj(self.ii_adj_visual)
            self.explicit_visual_adj = self.sparse_mx_to_torch_sparse_tensor(consensus_adj).float().to(self.device)
            zero_degree = int(np.count_nonzero(np.asarray(self.ii_adj_visual.sum(axis=1)).ravel() == 0))
            print('[Model Ablation] explicit_visual_graph=consensus; edges={}; zero_out_degree={}/{}'.format(
                self.ii_adj_visual.nnz, zero_degree, self.n_items))

        # 2.2 Text graph: KNN + P2B complement edges
        print("[Model] Building text graph (KNN + P2B)...")
        self.ii_adj_text = self.add_edge(self.inter, self.p2b_map)

        # 3) Normalize and convert to sparse tensors
        # Note: get_adj_mat defines self.R internally

        # 3.1 Visual Graph -> norm_adj_visual
        # get_adj_mat updates self.R from the visual graph, later overwritten by text graph
        norm_adj_visual_mat = self.get_adj_mat(self.ii_adj_visual)
        self.norm_adj_visual = self.sparse_mx_to_torch_sparse_tensor(norm_adj_visual_mat).float().to(self.device)
        visual_ui = self.R if self.visual_user_graph == 'own_visual' else None

        # 3.2 Text Graph -> norm_adj_text
        # This call updates self.R using the text graph
        norm_adj_text_mat = self.get_adj_mat(self.ii_adj_text)
        self.norm_adj_text = self.sparse_mx_to_torch_sparse_tensor(norm_adj_text_mat).float().to(self.device)

        # 3.3 Create R_tensor using the last updated self.R
        self.R_tensor = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        # Isolate the text-degree dependence of visual user-history aggregation.
        # This changes only the visual explicit UI block, not its edge support.
        self.visual_R_tensor = self.R_tensor
        if self.visual_user_graph == 'own_visual':
            self.visual_R_tensor = self.sparse_mx_to_torch_sparse_tensor(visual_ui).float().to(self.device)
            print('[Model Ablation] visual_user_graph=own_visual')

        # Layers
        self.iGUME_reduce_dim = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.iGUME_trans_dim = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.iGUME_space_trans = nn.Sequential(self.iGUME_reduce_dim, self.iGUME_trans_dim)

        self.text_reduce_dim = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.text_trans_dim = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.text_space_trans = nn.Sequential(self.text_reduce_dim, self.text_trans_dim)

        self.separate_coarse = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh(),
                                             nn.Linear(self.embedding_dim, 1, bias=False))
        self.softmax = nn.Softmax(dim=-1)
        self.iGUME_behavior = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.text_behavior = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.tau = 0.5

    def find_inter(self, iGUME_adj, text_adj):
        inter_file = os.path.join(self.dataset_path, 'inter.json')
        img_indices = iGUME_adj.coalesce()._indices().cpu()
        txt_indices = text_adj.coalesce()._indices().cpu()

        img_neighbors = defaultdict(set)
        txt_neighbors = defaultdict(set)

        for idx in range(img_indices.shape[1]):
            src = int(img_indices[0][idx].item())
            dst = int(img_indices[1][idx].item())
            img_neighbors[src].add(dst)

        for idx in range(txt_indices.shape[1]):
            src = int(txt_indices[0][idx].item())
            dst = int(txt_indices[1][idx].item())
            txt_neighbors[src].add(dst)

        all_ids = sorted(set(img_neighbors.keys()) | set(txt_neighbors.keys()))
        inter = {}
        for node_id in all_ids:
            common = sorted((img_neighbors.get(node_id, set()) & txt_neighbors.get(node_id, set())) - {node_id})
            if common:
                inter[node_id] = common

        with open(inter_file, "w") as f:
            json.dump(inter, f)
        return inter

    def add_edge(self, inter, p2b_map=None):
        P2B_WEIGHT = self.p2b_weight
        sim_rows = []
        sim_cols = []
        sim_values = []
        for id, vs in inter.items():
            if len(vs) == 0: continue
            for v in vs:
                sim_rows.append(int(id))
                sim_cols.append(v)
                sim_values.append(1.0)

        injected_count = 0
        if p2b_map:
            for src_str, targets in p2b_map.items():
                try:
                    src_id = int(src_str)
                except:
                    continue
                if not 0 <= src_id < self.n_items: continue
                current_targets = targets[:self.max_p2b_k]
                for tgt_str in current_targets:
                    try:
                        tgt_id = int(tgt_str)
                    except:
                        continue
                    if not 0 <= tgt_id < self.n_items or tgt_id == src_id: continue
                    sim_rows.append(src_id)
                    sim_cols.append(tgt_id)
                    sim_values.append(P2B_WEIGHT)
                    injected_count += 1
            print(f"[Model Info] P2B edge injection: {injected_count} edges added.")
        else:
            print("[Model Info] Clean KNN graph built (no P2B injected).")

        sim_rows = np.asarray(sim_rows, dtype=np.int64)
        sim_cols = np.asarray(sim_cols, dtype=np.int64)
        sim_values = np.array(sim_values, dtype=np.float32)
        item_adj = sp.coo_matrix((sim_values, (sim_rows, sim_cols)),
                                 shape=(self.n_items, self.n_items),
                                 dtype=np.float32)
        return item_adj

    def get_adj_mat(self, item_adj):
        # Block construction avoids the Python-level insertion and conversions of
        # DOK/LIL while retaining [[0, R], [R.T, item_adj]], including directed II edges.
        R = self.interaction_matrix.tocsr()
        adj_mat = sp.bmat([[None, R], [R.T, item_adj]], format='csr', dtype=np.float32)
        rowsum = np.asarray(adj_mat.sum(axis=1)).ravel()
        rowsum[rowsum <= 1e-9] = 1e-9
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv, format='csr')
        norm_adj_mat = (d_mat_inv @ adj_mat @ d_mat_inv).tocsr()
        # The text-graph call remains last, so shared explicit propagation still
        # uses the UI block normalized with the text graph's item degrees.
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat

    @staticmethod
    def normalize_item_adj(item_adj):
        """Item-only D^-1/2 A D^-1/2, preserving directed consensus edges.

        Zero out-degree nodes receive inverse degree zero, as in the existing
        KNN normalizer. Interaction degrees are intentionally excluded.
        """
        adjacency = item_adj.tocsr().astype(np.float32)
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inverse_sqrt = np.zeros_like(degree)
        nonzero = degree > 0
        inverse_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
        scale = sp.diags(inverse_sqrt, format='csr')
        return (scale @ adjacency @ scale).tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def conv_ui(self, adj, user_embeds, item_embeds):
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        return all_embeddings

    def conv_ii(self, ii_adj, single_modal):
        for i in range(self.n_layers):
            single_modal = torch.sparse.mm(ii_adj, single_modal)
        return single_modal

    # ============================================================
    # Forward: asymmetric propagation logic
    # ============================================================
    def forward(self, train=False):
        # 1) Feature transforms
        iGUME_item_embeds = torch.multiply(self.item_id_embedding.weight,
                                           self.iGUME_space_trans(self.iGUME_embedding.weight))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight,
                                          self.text_space_trans(self.text_embedding.weight))

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

        # 2) ID backbone propagation
        # Use the text graph (KNN + P2B) as the main backbone
        extended_id_embeds = self.conv_ui(self.norm_adj_text, user_embeds, item_embeds)

        # 3) Visual stream (asymmetric: KNN only)
        # Step 3.1: Explicit II (visual KNN)
        explicit_iGUME_item = self.conv_ii(self.explicit_visual_adj, iGUME_item_embeds)
        explicit_iGUME_user = torch.sparse.mm(self.visual_R_tensor, explicit_iGUME_item)
        explicit_iGUME_embeds = torch.cat([explicit_iGUME_user, explicit_iGUME_item], dim=0)

        # Step 3.2: Extended UI (visual graph, no P2B)
        extended_iGUME_embeds = self.conv_ui(self.norm_adj_visual, self.extended_iGUME_user.weight, explicit_iGUME_item)

        # 4) Text stream (asymmetric: KNN + P2B)
        # Step 4.1: Explicit II (text KNN)
        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R_tensor, explicit_text_item)
        explicit_text_embeds = torch.cat([explicit_text_user, explicit_text_item], dim=0)

        # Step 4.2: Extended UI (text graph with P2B)
        extended_text_embeds = self.conv_ui(self.norm_adj_text, self.extended_text_user.weight, explicit_text_item)

        # 5) Fusion
        extended_it_embeds = (extended_iGUME_embeds + extended_text_embeds) / 2

        # Coarse-grained
        img_score = self.separate_coarse(explicit_iGUME_embeds)
        txt_score = self.separate_coarse(explicit_text_embeds)

        iGUME_weights, text_weights = torch.split(self.softmax(
            torch.cat([img_score, txt_score], dim=-1)), 1, dim=-1)

        coarse_grained_embeds = iGUME_weights * explicit_iGUME_embeds + text_weights * explicit_text_embeds

        # Fine-grained
        fine_grained_iGUME = torch.multiply(self.iGUME_behavior(extended_id_embeds),
                                            (explicit_iGUME_embeds - coarse_grained_embeds))
        fine_grained_text = torch.multiply(self.text_behavior(extended_id_embeds),
                                           (explicit_text_embeds - coarse_grained_embeds))

        # BiLinear gating
        fine_grained_iGUME, fine_grained_text = self.bilinear_fusion(fine_grained_iGUME, fine_grained_text)

        # Final integration
        integration_embeds = fine_grained_iGUME + fine_grained_text + coarse_grained_embeds
        integration_embeds = integration_embeds / 3.0
        all_embeds = extended_id_embeds + integration_embeds

        if train:
            return all_embeds, (integration_embeds, extended_id_embeds, extended_it_embeds), (explicit_iGUME_embeds,
                                                                                              explicit_text_embeds)
        return all_embeds

    def sq_sum(self, emb):
        return 1. / 2 * (emb ** 2).sum()

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        regularizer = (self.sq_sum(users) + self.sq_sum(pos_items) + self.sq_sum(neg_items)) / self.batch_size
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        reg_loss = self.reg_weight_1 * regularizer
        return mf_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / (ttl_score + 1e-8))
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # Call forward without args (uses internal graphs)
        embeds_1, embeds_2, embeds_3 = self.forward(train=True)
        users_embeddings, items_embeddings = torch.split(embeds_1, [self.n_users, self.n_items], dim=0)

        integration_embeds, extended_id_embeds, extended_it_embeds = embeds_2
        explicit_iGUME_embeds, explicit_text_embeds = embeds_3

        u_g_embeddings = users_embeddings[users]
        pos_i_g_embeddings = items_embeddings[pos_items]
        neg_i_g_embeddings = items_embeddings[neg_items]

        # VT Loss
        vt_loss = self.vt_loss * self.align_vt(explicit_iGUME_embeds, explicit_text_embeds)

        integration_users, integration_items = torch.split(integration_embeds, [self.n_users, self.n_items], dim=0)
        extended_id_user, extended_id_items = torch.split(extended_id_embeds, [self.n_users, self.n_items], dim=0)

        # BPR Loss
        bpr_loss, reg_loss_1 = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        # BM Loss
        loss_bm_user = self.InfoNCE(integration_users[users], extended_id_user[users], self.bm_temp)
        loss_bm_item = self.InfoNCE(integration_items[pos_items], extended_id_items[pos_items], self.bm_temp)
        bm_loss = self.bm_loss * (loss_bm_user + loss_bm_item)

        al_loss = vt_loss + bm_loss

        extended_it_user, extended_it_items = torch.split(extended_it_embeds, [self.n_users, self.n_items], dim=0)
        c_loss = self.InfoNCE(extended_it_user[users], integration_users[users], self.um_temp)
        noise_loss_1 = self.cal_noise_loss(users, integration_users, self.um_temp)
        noise_loss_2 = self.cal_noise_loss(users, extended_it_user, self.um_temp)

        um_loss = self.um_loss * (c_loss + noise_loss_1 + noise_loss_2)

        reg_loss_2 = self.reg_weight_2 * self.sq_sum(extended_it_items[pos_items]) / self.batch_size
        reg_loss = reg_loss_1 + reg_loss_2

        total_loss = bpr_loss + al_loss + um_loss + reg_loss
        # Detached values let the trainer sample diagnostics without retaining
        # the autograd graph or introducing CPU/GPU synchronization each batch.
        self.last_loss_components = {
            'bpr': bpr_loss.detach(),
            'bm': bm_loss.detach(),
            'um': um_loss.detach(),
            'vt': vt_loss.detach(),
            'reg': reg_loss.detach(),
            'total': total_loss.detach(),
        }

        return total_loss

    def cal_noise_loss(self, id, emb, temp):
        def add_perturbation(x):
            random_noise = torch.rand_like(x).to(self.device)
            x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
            return x

        emb_view1 = add_perturbation(emb)
        emb_view2 = add_perturbation(emb)
        emb_loss = self.InfoNCE(emb_view1[id], emb_view2[id], temp)
        return emb_loss

    def align_vt(self, embed1, embed2):
        emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
        emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)
        vt_loss = (torch.abs(emb1_var - emb2_var) + torch.abs(emb1_mean - emb2_mean)).mean()
        return vt_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        # Call forward without args
        all_embeds = self.forward(train=False)
        restore_user_e, restore_item_e = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

