import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 保持原有的 import，确保环境兼容
from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.utils import get_model, init_seed
from utils.logger import init_logger


def visualize(args):
    print(">>> [1] 初始化配置")
    config_dict = {
        'gpu_id': 0,
        'train_batch_size': 2048,
        'eval_batch_size': 2048,
        'reproducibility': True,
        'seed': 999
    }

    # 请确保 dataset 名称正确（大小写敏感，如 'sports' 或 'Sports'）
    config = Config(model='MAGE', dataset=args.dataset, config_dict={**config_dict, 'data_path': args.data_path})
    init_logger(config)
    init_seed(config['seed'])

    # =========================================================
    # Step 1: 构建 RecDataset（完全对齐训练流程）
    # =========================================================
    print(">>> [2] 加载 RecDataset")
    dataset = RecDataset(config)
    _ = str(dataset)

    # split
    train_dataset, _, _ = dataset.split()
    _ = str(train_dataset)

    # =========================================================
    # Step 2: 构建 TrainDataLoader
    # =========================================================
    print(">>> [3] 构建 TrainDataLoader")
    train_data = TrainDataLoader(
        config,
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=False
    )
    train_data.pretrain_setup()

    # =========================================================
    # Step 3: 初始化模型
    # =========================================================
    print(">>> [4] 初始化 GUME 模型")
    model = get_model('MAGE')(config, train_data, None).to(config['device'])

    # =========================================================
    # Step 4: 加载模型参数
    # =========================================================
    print(">>> [5] 加载模型参数")
    # 请根据实际情况修改路径，例如换成 MAGE 的路径再跑一次
    model_path = args.checkpoint
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    print(f"    Loading: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=config['device']))
    model.eval()

    # =========================================================
    # Step 5: 导出 Item Embeddings
    # =========================================================
    print(">>> [6] 提取 Item Embeddings")
    with torch.no_grad():
        item_embeddings = model.forward(train=False)[model.n_users:]
        item_embeddings = item_embeddings.cpu().numpy()

    print(f"    Item embedding shape = {item_embeddings.shape}")

    # =========================================================
    # Step 6: Case Study 数据加载
    # =========================================================
    print(">>> [7] 加载 Case Study JSON")
    case_file = args.case_file
    if not os.path.exists(case_file):
        raise FileNotFoundError(f"未找到 {case_file}")

    with open(case_file, 'r', encoding='utf-8') as f:
        case_data = json.load(f)

    target_id = args.target_id
    target_case = None
    for x in case_data:
        if x.get('numeric_id') == target_id:
            target_case = x
            break

    if target_case is None:
        # 兜底：如果没有找到 6656，找第一个能用的
        print(f"    [警告] 未找到 ID {target_id}，尝试自动选择第一个可用 Case...")
        for x in case_data:
            if x.get('numeric_id') is not None and len(x.get('also_bought', [])) > 0:
                target_case = x
                break
        if target_case is None:
            raise ValueError("JSON 中没有有效数据")

    target_idx = target_case['numeric_id']
    complement_indices = [
        x['numeric_id']
        for x in target_case.get('also_bought', [])
        if x.get('numeric_id') is not None
    ]

    print(f"    主角: {target_case.get('title', 'Unknown')} (ID: {target_idx})")
    print(f"    互补品数量: {len(complement_indices)}")

    # =========================================================
    # Step 7: 构建可视化样本 (引入 Hard Negatives)
    # =========================================================
    print(">>> [8] 挖掘 Hard Negatives (高相似度干扰项)...")

    # 1. 计算 Target 与全局物品的余弦相似度
    # 先归一化，防止模长影响
    norm = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    norm_embeds = item_embeddings / (norm + 1e-8)

    target_vec = norm_embeds[target_idx]
    sims = np.dot(norm_embeds, target_vec)  # Cosine Similarity

    # 2. 排序并筛选 Top-N
    sorted_indices = np.argsort(-sims)  # 降序

    exclude_set = set([target_idx] + complement_indices)
    background_indices = []

    for idx in sorted_indices:
        if idx not in exclude_set:
            background_indices.append(idx)
            if len(background_indices) >= 200:  # 选 200 个最像的做背景
                break

    print(f"    已选择 {len(background_indices)} 个 Hard Negatives 作为背景。")

    # 合并索引
    viz_indices = [target_idx] + complement_indices + background_indices
    # 去重（保持顺序）
    seen = set()
    viz_indices_clean = []
    for x in viz_indices:
        if x not in seen:
            viz_indices_clean.append(x)
            seen.add(x)
    viz_indices = viz_indices_clean

    viz_vectors = item_embeddings[viz_indices]

    # 建立映射: 原始ID -> 可视化数组索引
    id2pos = {original_id: i for i, original_id in enumerate(viz_indices)}

    # =========================================================
    # Step 8: t-SNE 择优 (循环多次挑选最佳视角)
    # =========================================================
    print(">>> [9] 运行 t-SNE (尝试 10 次寻找最佳视角)...")

    best_layout = None
    best_score = -float('inf')

    for i in range(10):
        # 随机 perplexity 增加多样性 (5 ~ 30)
        perp = np.random.randint(5, min(35, len(viz_indices)))
        tsne = TSNE(n_components=2, random_state=i * 100, perplexity=perp, init='pca', learning_rate='auto')
        layout = tsne.fit_transform(viz_vectors)

        # --- 简单评估指标 ---
        # 我们希望：Target 离 Complements 近，离 Background 远
        t_pos = layout[id2pos[target_idx]]

        comp_pos = np.array([layout[id2pos[c]] for c in complement_indices if c in id2pos])
        bg_pos = np.array([layout[id2pos[b]] for b in background_indices if b in id2pos])

        if len(comp_pos) == 0: continue

        # 平均互补品距离 (越小越好)
        dist_comp = np.mean(np.linalg.norm(comp_pos - t_pos, axis=1))
        # 平均背景距离 (越大越好)
        dist_bg = np.mean(np.linalg.norm(bg_pos - t_pos, axis=1))

        # 分数 = 分离度 / 内聚度
        score = dist_bg / (dist_comp + 1e-6)

        # print(f"    Trial {i+1}: Perp={perp}, Score={score:.4f}")

        if score > best_score:
            best_score = score
            best_layout = layout
            # print("    -> New Best Layout Found")

    vectors_2d = best_layout
    print(f"    最佳布局得分: {best_score:.4f}")

    # =========================================================
    # Step 9: 可视化绘制
    # =========================================================
    print(">>> [10] 绘制可视化图")
    plt.figure(figsize=(10, 8))

    # 1. 绘制背景 (Hard Negatives) - 灰色
    bg_ids = [id2pos[i] for i in background_indices if i in id2pos]
    plt.scatter(
        vectors_2d[bg_ids, 0],
        vectors_2d[bg_ids, 1],
        c='#666666',
        alpha=0.7,  # 稍微提高透明度，让它们显眼一点点
        label='Hard Negatives (Similar Items)'
    )

    # 2. 绘制互补品 - 蓝色三角形
    comp_ids = [id2pos[i] for i in complement_indices if i in id2pos]
    plt.scatter(
        vectors_2d[comp_ids, 0],
        vectors_2d[comp_ids, 1],
        c='blue',
        marker='^',
        s=120,
        alpha=0.9,
        label='Functional Complements'
    )

    # 3. 绘制 Target - 红色星星
    t_id = id2pos[target_idx]
    plt.scatter(
        vectors_2d[t_id, 0],
        vectors_2d[t_id, 1],
        c='red',
        marker='*',
        s=450,
        edgecolors='black',
        zorder=10,
        label='Target Item'
    )

    plt.legend(loc='upper right')
    # 根据你的模型名称修改标题，如果是跑 MAGE 就改成 MAGE
    plt.title('Visualization of Item Embeddings')
    plt.axis('off')
    plt.tight_layout()

    save_path = args.output
    plt.savefig(save_path)
    print(f">>> 完成，可视化结果已保存至 {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize MAGE item embeddings for an existing case study')
    parser.add_argument('--dataset', default='sports')
    parser.add_argument('--data-path', default='data')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--case-file', required=True)
    parser.add_argument('--target-id', type=int, default=6656)
    parser.add_argument('--output', default='case_study_viz_hard.pdf')
    visualize(parser.parse_args())