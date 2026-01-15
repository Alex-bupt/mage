import os
import sys
import json
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_knn_normalized_graph


# =============================================================================
# [DEBUG TOOL] Numerical anomaly checker
# =============================================================================
def check_tensor(name, tensor, stop_on_error=True):
    """
    Check a tensor for NaN/Inf values.

    Args:
        name: Label for error reporting.
        tensor: Tensor to check.
        stop_on_error: If True, terminate the process on detection.

    Returns:
        True if NaN/Inf is found, otherwise False.
    """
    if tensor is None:
        return False
    if torch.isnan(tensor).any():
        print(f"\n[CRITICAL ERROR] Found NaN in: {name}")
        if stop_on_error:
            sys.exit(1)
        return True
    if torch.isinf(tensor).any():
        print(f"\n[CRITICAL ERROR] Found Inf in: {name}")
        if stop_on_error:
            sys.exit(1)
        return True
    return False


# =============================================================================
# BiLinearGate
# =============================================================================
class BiLinearGate(nn.Module):
    """
    A symmetric bi-linear gating module that exchanges gating signals
    across two modalities (e.g., visual vs. textual streams).
    """
    def __init__(self, dim, reduction_ratio=8):
        super(BiLinearGate, self).__init__()
        hidden_dim = max(dim // reduction_ratio, 16)

        self.gate_v_maker = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.v_self_calib = nn.Linear(dim, dim, bias=False)

        self.gate_t_maker = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.t_self_calib = nn.Linear(dim, dim, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_v, x_t):
        """
        Args:
            x_v: Visual stream embedding (N x D).
            x_t: Text stream embedding (N x D).

        Returns:
            Gated visual embedding and gated text embedding.
        """
        suggestion_from_text = self.gate_t_maker(x_t)
        defense_from_visual = self.v_self_calib(x_v)
        g_v = self.sigmoid(suggestion_from_text + defense_from_visual)

        suggestion_from_visual = self.gate_v_maker(x_v)
        defense_from_text = self.t_self_calib(x_t)
        g_t = self.sigmoid(suggestion_from_visual + defense_from_text)

        x_v_gated = x_v * g_v
        x_t_gated = x_t * g_t
        return x_v_gated, x_t_gated


# =============================================================================
# Model: Asymmetric Dual-Stream Graph Recommender (anonymized naming)
# =============================================================================
class MAGE(GeneralRecommender):
    """
    A dual-stream recommender with asymmetric graph propagation:
    - ID backbone uses a richer (text-side) graph
    - Visual stream uses a "clean" KNN-only graph
    - Text stream uses KNN plus an optional injected edge map
    - Two streams are fused by coarse/fine integration + bi-linear gating

    Note:
    This file keeps the original class name for compatibility with your codebase.
    For strict double-blind release, see the anonymity checklist at the end.
    """
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

        # Trainable ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Trainable user parameters used by the two extended streams
        self.extended_iMAGE_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_iMAGE_user.weight)
        self.extended_text_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_text_user.weight)

        # Dataset root (path is constructed from config; ensure config values are anonymized)
        self.dataset_path = os.path.abspath(os.getcwd() + config['data_path'] + config['dataset'])

        self.bilinear_fusion = BiLinearGate(self.embedding_dim)

        # Load optional injected edge map (external complement edges)
        self.p2b_map = {}
        injected_map_file = os.path.join(self.dataset_path, 'p2b_map_filtered.json')
        if os.path.exists(injected_map_file):
            with open(injected_map_file, 'r', encoding='utf-8') as f:
                self.p2b_map = json.load(f)

        # Cached adjacency files
        image_adj_file = os.path.join(self.dataset_path, f'iMAGE_adj_{self.knn_k}_{self.sparse}.pt')
        text_adj_file = os.path.join(self.dataset_path, f'text_adj_{self.knn_k}_{self.sparse}.pt')

        # 1) Load raw features and build KNN graphs (or load cached)
        if self.v_feat is not None:
            self.iMAGE_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.iMAGE_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(
                    image_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym'
                )
                torch.save(image_adj, image_adj_file)
            self.iMAGE_original_adj = image_adj.to(self.device)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(
                    text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym'
                )
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.to(self.device)

        # 2) Build intersection structure between two KNN graphs (precomputed if exists)
        self.inter = self.find_inter(self.iMAGE_original_adj, self.text_original_adj)

        # 2.1 Visual-only item graph: KNN intersection edges only (no injection)
        print("[Model] Building visual item graph (KNN only)...")
        self.ii_adj_visual = self.add_edge(self.inter, injected_map=None)

        # 2.2 Text-side item graph: KNN intersection + injected complement edges (if provided)
        print("[Model] Building text item graph (KNN + injected edges)...")
        self.ii_adj_text = self.add_edge(self.inter, injected_map=self.p2b_map)

        # 3) Normalize UI+II graph and convert to torch sparse tensors
        # 3.1 Visual graph normalization
        norm_adj_visual_mat = self.get_adj_mat(self.ii_adj_visual.tolil())
        self.norm_adj_visual = self.sparse_mx_to_torch_sparse_tensor(norm_adj_visual_mat).float().to(self.device)

        # 3.2 Text graph normalization (this call updates self.R)
        norm_adj_text_mat = self.get_adj_mat(self.ii_adj_text.tolil())
        self.norm_adj_text = self.sparse_mx_to_torch_sparse_tensor(norm_adj_text_mat).float().to(self.device)

        # 3.3 Store R tensor for explicit propagation (uses latest self.R)
        self.R_tensor = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)

        # Feature projection layers
        self.iMAGE_reduce_dim = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.iMAGE_trans_dim = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.iMAGE_space_trans = nn.Sequential(self.iMAGE_reduce_dim, self.iMAGE_trans_dim)

        self.text_reduce_dim = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.text_trans_dim = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.text_space_trans = nn.Sequential(self.text_reduce_dim, self.text_trans_dim)

        # Fusion modules
        self.separate_coarse = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.iMAGE_behavior = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.text_behavior = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())

        # Unused in the provided snippet but kept for compatibility
        self.tau = 0.5

    def find_inter(self, iMAGE_adj, text_adj):
        """
        Compute (or load) an item-wise intersection of the neighbor sets implied by
        the two modality-specific KNN graphs.

        The persisted file name is generic; adjust for strict anonymity.
        """
        inter_file = os.path.join(self.dataset_path, 'inter.json')
        if os.path.exists(inter_file):
            with open(inter_file) as f:
                inter = json.load(f)
        else:
            j = 0
            inter = defaultdict(list)
            img_sim, txt_sim = [], []

            img_indices = iMAGE_adj._indices().cpu()
            txt_indices = text_adj._indices().cpu()

            for i in range(0, len(img_indices[0])):
                img_id = img_indices[0][i]
                txt_id = txt_indices[0][i]
                assert img_id == txt_id

                item_id = img_id.item()
                img_sim.append(img_indices[1][j].item())
                txt_sim.append(txt_indices[1][j].item())

                # NOTE: hard-coded "10" is derived from the way indices are iterated; keep consistent with graph builder
                if len(img_sim) == 10 and len(txt_sim) == 10:
                    inter_set = list(set(img_sim) & set(txt_sim))
                    inter[item_id] = [v for v in inter_set if v != item_id]
                    img_sim, txt_sim = [], []
                j += 1

            with open(inter_file, "w") as f:
                json.dump(inter, f)
        return inter

    def add_edge(self, inter, injected_map=None):
        """
        Build an item-item adjacency matrix from:
        - base edges: inter (intersection neighbors, weight=1.0)
        - optional injected edges: injected_map (weight=PSEUDO_EDGE_WEIGHT)

        Args:
            inter: dict[item_id -> list[neighbor_item_id]]
            injected_map: dict[str(item_id) -> list[str(neighbor_item_id)]] or None

        Returns:
            scipy.sparse.coo_matrix adjacency of shape (n_items, n_items)
        """
        MAX_INJECT_K = 20
        PSEUDO_EDGE_WEIGHT = 0.4

        sim_rows, sim_cols, sim_values = [], [], []

        for item_id, neighs in inter.items():
            if len(neighs) == 0:
                continue
            for n in neighs:
                sim_rows.append(int(item_id))
                sim_cols.append(int(n))
                sim_values.append(1.0)

        injected_count = 0
        if injected_map:
            for src_str, targets in injected_map.items():
                try:
                    src_id = int(src_str)
                except Exception:
                    continue
                if src_id >= self.n_items:
                    continue

                for tgt_str in targets[:MAX_INJECT_K]:
                    try:
                        tgt_id = int(tgt_str)
                    except Exception:
                        continue
                    if tgt_id >= self.n_items or tgt_id == src_id:
                        continue

                    sim_rows.append(src_id)
                    sim_cols.append(tgt_id)
                    sim_values.append(PSEUDO_EDGE_WEIGHT)
                    injected_count += 1

            print(f"[Model Info] Injected edges added: {injected_count}")
        else:
            print("[Model Info] Built clean KNN graph (no injected edges).")

        sim_rows_t = torch.tensor(sim_rows, dtype=torch.int64)
        sim_cols_t = torch.tensor(sim_cols, dtype=torch.int64)
        sim_values_np = np.asarray(sim_values, dtype=np.float32)

        item_adj = sp.coo_matrix(
            (sim_values_np, (sim_rows_t, sim_cols_t)),
            shape=(self.n_items, self.n_items),
            dtype=np.float32,
        )
        return item_adj

    def get_adj_mat(self, item_adj):
        """
        Construct a unified adjacency matrix:
        [ U-U   U-I ]
        [ I-U   I-I ]
        where U-I is the user-item interaction matrix, and I-I is item_adj.

        Returns:
            Row/col symmetrically normalized adjacency matrix (CSR).
        Side effect:
            Updates self.R for explicit propagation (U x I block of normalized adjacency).
        """
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32
        ).tolil()

        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat[self.n_users:, self.n_users:] = item_adj
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            rowsum[rowsum <= 1e-9] = 1e-9
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)

            # IMPORTANT: Uses adj_mat from the outer scope (original behavior retained)
            norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat).tolil()

        # Store U-I block for explicit user propagation
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    @staticmethod
    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse FloatTensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def conv_ui(self, adj, user_embeds, item_embeds):
        """
        UI-propagation over a unified (U+I) graph via sparse matrix multiplication.
        Uses mean aggregation over (0..n_ui_layers) hops.
        """
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1, keepdim=False)
        return all_embeddings

    def conv_ii(self, ii_adj, single_modal):
        """Item-only propagation on an item-item graph for a single modality embedding."""
        for _ in range(self.n_layers):
            single_modal = torch.sparse.mm(ii_adj, single_modal)
        return single_modal

    # =============================================================================
    # Forward: asymmetric propagation logic
    # =============================================================================
    def forward(self, train=False):
        # 1) Feature transformation (modality embeddings are gated by item ID embedding)
        iMAGE_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.iMAGE_space_trans(self.iMAGE_embedding.weight),
        )
        text_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.text_space_trans(self.text_embedding.weight),
        )

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

        # 2) ID backbone propagation (uses richer graph by design)
        extended_id_embeds = self.conv_ui(self.norm_adj_text, user_embeds, item_embeds)

        # 3) Visual stream (KNN-only graph)
        explicit_iMAGE_item = self.conv_ii(self.iMAGE_original_adj, iMAGE_item_embeds)
        explicit_iMAGE_user = torch.sparse.mm(self.R_tensor, explicit_iMAGE_item)
        explicit_iMAGE_embeds = torch.cat([explicit_iMAGE_user, explicit_iMAGE_item], dim=0)

        extended_iMAGE_embeds = self.conv_ui(
            self.norm_adj_visual,
            self.extended_iMAGE_user.weight,
            explicit_iMAGE_item,
        )

        # 4) Text stream (KNN + injected edges)
        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R_tensor, explicit_text_item)
        explicit_text_embeds = torch.cat([explicit_text_user, explicit_text_item], dim=0)

        extended_text_embeds = self.conv_ui(
            self.norm_adj_text,
            self.extended_text_user.weight,
            explicit_text_item,
        )

        # 5) Fusion
        extended_it_embeds = (extended_iMAGE_embeds + extended_text_embeds) / 2.0

        # Coarse-grained fusion: learn weights per node between explicit modality embeddings
        img_score = self.separate_coarse(explicit_iMAGE_embeds)
        txt_score = self.separate_coarse(explicit_text_embeds)

        iMAGE_w, text_w = torch.split(
            self.softmax(torch.cat([img_score, txt_score], dim=-1)),
            1,
            dim=-1,
        )
        coarse_embeds = iMAGE_w * explicit_iMAGE_embeds + text_w * explicit_text_embeds

        # Fine-grained fusion: behavior-conditioned residuals
        fine_iMAGE = torch.multiply(
            self.iMAGE_behavior(extended_id_embeds),
            (explicit_iMAGE_embeds - coarse_embeds),
        )
        fine_text = torch.multiply(
            self.text_behavior(extended_id_embeds),
            (explicit_text_embeds - coarse_embeds),
        )

        # Bi-linear gating between fine-grained components
        fine_iMAGE, fine_text = self.bilinear_fusion(fine_iMAGE, fine_text)

        # Final integration
        integration_embeds = (fine_iMAGE + fine_text + coarse_embeds) / 3.0
        all_embeds = extended_id_embeds + integration_embeds

        if train:
            return (
                all_embeds,
                (integration_embeds, extended_id_embeds, extended_it_embeds),
                (explicit_iMAGE_embeds, explicit_text_embeds),
            )
        return all_embeds

    @staticmethod
    def sq_sum(emb):
        """Half squared L2 sum (used as a standard embedding regularizer)."""
        return 0.5 * (emb ** 2).sum()

    def bpr_loss(self, users, pos_items, neg_items):
        """
        Bayesian Personalized Ranking (BPR) loss with L2 regularization.
        """
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)

        regularizer = (self.sq_sum(users) + self.sq_sum(pos_items) + self.sq_sum(neg_items)) / self.batch_size
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        reg_loss = self.reg_weight_1 * regularizer
        return mf_loss, reg_loss

    @staticmethod
    def InfoNCE(view1, view2, temperature):
        """
        InfoNCE contrastive loss (within-batch negatives).
        """
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)

        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)

        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

        cl_loss = -torch.log(pos_score / (ttl_score + 1e-8))
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        """
        Total training loss:
        - BPR loss on final embeddings
        - Alignment losses between streams (variance/mean + contrastive)
        - User-modeling and noise-robustness contrastive losses
        - L2 regularization terms
        """
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]

        embeds_all, aux_pack, explicit_pack = self.forward(train=True)
        users_embeddings, items_embeddings = torch.split(embeds_all, [self.n_users, self.n_items], dim=0)

        integration_embeds, extended_id_embeds, extended_it_embeds = aux_pack
        explicit_iMAGE_embeds, explicit_text_embeds = explicit_pack

        u_g = users_embeddings[users]
        pos_i_g = items_embeddings[pos_items]
        neg_i_g = items_embeddings[neg_items]

        # Stream alignment loss (mean/variance matching)
        vt_loss = self.vt_loss * self.align_vt(explicit_iMAGE_embeds, explicit_text_embeds)

        integration_users, integration_items = torch.split(integration_embeds, [self.n_users, self.n_items], dim=0)
        extended_id_users, extended_id_items = torch.split(extended_id_embeds, [self.n_users, self.n_items], dim=0)

        # BPR loss on final representations
        bpr_loss, reg_loss_1 = self.bpr_loss(u_g, pos_i_g, neg_i_g)

        # Backbone matching (contrastive)
        loss_bm_user = self.InfoNCE(integration_users[users], extended_id_users[users], self.bm_temp)
        loss_bm_item = self.InfoNCE(integration_items[pos_items], extended_id_items[pos_items], self.bm_temp)
        bm_loss = self.bm_loss * (loss_bm_user + loss_bm_item)

        al_loss = vt_loss + bm_loss

        # User modeling / cross-view consistency
        extended_it_users, extended_it_items = torch.split(extended_it_embeds, [self.n_users, self.n_items], dim=0)
        c_loss = self.InfoNCE(extended_it_users[users], integration_users[users], self.um_temp)

        # Noise-robustness losses
        noise_loss_1 = self.cal_noise_loss(users, integration_users, self.um_temp)
        noise_loss_2 = self.cal_noise_loss(users, extended_it_users, self.um_temp)

        um_loss = self.um_loss * (c_loss + noise_loss_1 + noise_loss_2)

        # Additional regularization
        reg_loss_2 = self.reg_weight_2 * self.sq_sum(extended_it_items[pos_items]) / self.batch_size
        reg_loss = reg_loss_1 + reg_loss_2

        total_loss = bpr_loss + al_loss + um_loss + reg_loss
        return total_loss

    def cal_noise_loss(self, idx, emb, temp):
        """
        Two-view perturbation-based InfoNCE (sign-directed normalized noise).
        """
        def add_perturbation(x):
            random_noise = torch.rand_like(x).to(self.device)
            x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
            return x

        view1 = add_perturbation(emb)
        view2 = add_perturbation(emb)
        return self.InfoNCE(view1[idx], view2[idx], temp)

    @staticmethod
    def align_vt(embed1, embed2):
        """
        Match first and second moments (mean/variance) between two embedding sets.
        """
        emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
        emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)
        return (torch.abs(emb1_var - emb2_var) + torch.abs(emb1_mean - emb2_mean)).mean()

    def full_sort_predict(self, interaction):
        """
        Full ranking over all items for a given user index.
        """
        user = interaction[0]
        all_embeds = self.forward(train=False)
        user_e, item_e = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        u_e = user_e[user]
        scores = torch.matmul(u_e, item_e.transpose(0, 1))
        return scores
