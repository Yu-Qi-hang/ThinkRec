import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import pandas as pd
import json
import numpy as np
# from minigpt4.models.rec_model import disabled_train
from minigpt4.models.rec_base_models import MatrixFactorization, MF_linear,LightGCN, SASRec, Personlized_Prompt, random_mf, Soft_Prompt, RecEncoder_DIN
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings
import argparse

class Rec_config:
    def __init__(self, para_dict):
        self.user_num = para_dict.get("user_num", -100)
        self.item_num = para_dict.get("item_num", -100)
    def add_para_mf(self, para_dict):
        self.embedding_size = para_dict.get("embedding_size", 256)
    def add_para_l(self, para_dict):
        self.embedding_size = para_dict.get("embedding_size", 256)
        self.embed_size = para_dict.get("embed_size", 64)
        self.gcn_layers = para_dict.get("gcn_layers", 2)
        self.dropout = para_dict.get("dropout", False)
        self.keep_prob = para_dict.get("keep_prob", 0.6)
        self.A_n_fold = para_dict.get("A_n_fold", 100)
        self.A_split = para_dict.get("A_split", False)
        self.pretrain = para_dict.get("pretrain", 0)
        self.init_emb = para_dict.get("init_emb", 1e-1)
        self.dataset = para_dict.get("dataset", None)
    def add_para_s(self, para_dict):
        self.hidden_units = para_dict.get("hidden_units", 64)
        self.dropout_rate = para_dict.get("dropout_rate", 0.2)
        self.maxlen = para_dict.get("maxlen", 25)
        self.l2_emb = para_dict.get("l2_emb", 1e-4)
        self.num_blocks = para_dict.get("num_blocks", 2)
        self.num_heads = para_dict.get("num_heads", 1)

def init_rec_encoder(rec_model, config):
    if rec_model == "MF":
        print("### rec_encoder:", "MF")
        rec_model = MatrixFactorization(config)
    elif rec_model == "lightgcn":
        print("### rec_encoder:", "lightgcn")
        rec_model = LightGCN(config)
    elif rec_model == "sasrec":
        print("### rec_encoder:", "sasrec")
        rec_model = SASRec(config)
    elif rec_model == "DIN":
        print("### rec_encoder:", "DIN")
        rec_model = RecEncoder_DIN(config)
    else:
        rec_model = None
        warnings.warn(" the input rec_model is not MF, LightGCN or sasrec, or DCN, we won't utilize the rec_encoder directly.")
    return rec_model

# 方法1: K-Means聚类
def kmeans_clustering(data, n_clusters=None):
    if n_clusters is None or n_clusters<0:
        silhouette_scores = []
        cluster_range = range(2, 11)
        
        for n_clusters in tqdm(cluster_range):
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"最佳聚类数: {best_n_clusters}")
    else:
        best_n_clusters = n_clusters
    final_kmeans = MiniBatchKMeans(n_clusters=best_n_clusters, random_state=42, batch_size=1024, n_init=10)
    return best_n_clusters,final_kmeans.fit_predict(data)

# 方法2: 层次聚类
# def hierarchical_clustering(data, n_clusters=None):
#     if n_clusters is None or n_clusters<0:
#         silhouette_scores = []
#         cluster_range = range(2, 11)
        
#         for n_clusters in tqdm(cluster_range):
#             kmeans = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
#             cluster_labels = kmeans.fit_predict(data)
#             silhouette_avg = silhouette_score(data, cluster_labels)
#             silhouette_scores.append(silhouette_avg)
#         best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
#         print(f"最佳聚类数: {best_n_clusters}")
#     else:
#         best_n_clusters = n_clusters
#     final_kmeans = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
#     return best_n_clusters,final_kmeans.fit_predict(data)
def hierarchical_clustering(data, n_clusters=5):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    labels = clustering.fit_predict(data)
    return labels

# 使用k-距离曲线确定最佳eps（常用方法）
def find_optimal_eps(data, k=5):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data)
    distances, _ = neigh.kneighbors(data)
    k_distances = np.sort(distances[:, -1])
    
    plt.plot(k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.savefig('k_distance_curve.png')
    return np.percentile(k_distances, 95)  # 通常取曲线拐点附近的值

# 方法3: DBSCAN聚类 (适用于密度不均匀的数据)
def dbscan_clustering(data, dim=256):
    optimal_eps = 23000#find_optimal_eps(data, k=dim+1)  # k通常取embedding维度+1
    print(f"使用eps={optimal_eps}进行DBSCAN聚类")
    dbscan = DBSCAN(eps=optimal_eps, min_samples=2*dim)  # min_samples通常取维度*2或更大
    labels = dbscan.fit_predict(embeddings_scaled)
    print(f"自动发现的簇数: {len(np.unique(labels)) - (1 if -1 in labels else 0)}")
    print(f"噪声点数量: {np.sum(labels == -1)}")
    return labels

# # 可视化 (使用t-SNE降维到2D)
def visualize_clusters(embeddings, labels, save_fig):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(16, 8))
    # 设置透明背景
    fig = plt.gcf()
    fig.patch.set_facecolor('white')  # 或者使用 'none' 表示完全透明
    fig.patch.set_alpha(0)  
    # s_values = np.ones_like(embeddings_2d[:, 0]) * 50
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6,s=200)
    # plt.tick_params(axis='both', which='major', labelsize=20)
    # 显示横纵坐标轴（spines）
    ax = plt.gca()  # 获取当前坐标轴
    # 去除背景颜色
    ax.set_facecolor("white")  # 或 "none" 如果你想透明
    ax.patch.set_alpha(0)       # 背景透明
    # 去除网格线
    ax.grid(False)
    ax.spines['top'].set_visible(True)     # 显示顶部边框
    ax.spines['right'].set_visible(True)   # 显示右侧边框
    ax.spines['bottom'].set_visible(True)  # 显示底部边框
    ax.spines['left'].set_visible(True)    # 显示左侧边框
    # 可选：设置坐标轴粗细（宽度）
    ax.spines['top'].set_linewidth(8)
    ax.spines['right'].set_linewidth(8)
    ax.spines['bottom'].set_linewidth(8)
    ax.spines['left'].set_linewidth(8)
    # 设置刻度线更粗，同时隐藏刻度标签
    plt.tick_params(
        axis='both',       # 对x轴和y轴都生效
        which='both',      # 同时包括主刻度和次刻度
        direction='in',    # 刻度线朝内
        length=10,          # 刻度线长度
        width=5,           # 刻度线粗细
        labelleft=False,   # 不显示 y 轴刻度标签
        labelbottom=False  # 不显示 x 轴刻度标签
    )
    # plt.colorbar(scatter)
    # plt.title('t-SNE Visualization of User Clusters')
    # plt.xlabel('t-SNE 1')
    # plt.ylabel('t-SNE 2')
    # plt.savefig(save_fig)
    plt.tight_layout()  # 自动调整布局防止裁剪
    plt.savefig(save_fig, transparent=True, bbox_inches='tight')  # 保存时背景透明
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/home/yuqihang/projects/CoLLM/collm-datasets/bookdu")
    parser.add_argument('--mode', type=str, default="h")
    parser.add_argument('--n', type=int, default=-1)
    parser.add_argument('--pretrained_rec', type=str, default="/data/yuqihang/result/CoLLM/checkpoints/mf/0228_booknew_best_model_d256_lr0.001_wd1e-06.pth")
    # parser.add_argument('--data_ref', type=str, default="/home/yuqihang/projects/CoLLM/collm-datasets/booknew")
    args = parser.parse_args()
    data_dir = args.data_dir
    if data_dir[-2:] == 'du':
        args.data_ref = data_dir[:-2]+'new'
    if not os.path.exists(args.data_ref):
        args.data_ref = data_dir[:-2]
    n_clusters = args.n
    if 'lightgcn' in args.pretrained_rec:
        rec_model = "lightgcn"
    elif 'sasrec' in args.pretrained_rec:
        rec_model = "sasrec"
    else:
        rec_model = "MF"
    pretrained_rec = args.pretrained_rec

    train_ = pd.read_pickle(os.path.join(data_dir,"train_ood2.pkl"))
    valid_ = pd.read_pickle(os.path.join(data_dir,"valid_ood2.pkl"))
    test_ = pd.read_pickle(os.path.join(data_dir,"test_ood2.pkl"))
    total_ = pd.concat([train_[['uid','iid', 'his']],valid_[['uid','iid', 'his']],test_[['uid','iid', 'his']]],axis=0)
    reason_ = pd.read_pickle(os.path.join(data_dir,"reason_ood2.pkl"))
    user_num = total_.uid.max()+1
    item_num = total_.iid.max()+1

    print('Loading Rec_model')
    if rec_model == "MF":
        rec_config_ = {'user_num':int(user_num), 'item_num':int(item_num), 'embedding_size':256}
        rec_config = Rec_config(rec_config_)
        rec_config.add_para_mf(rec_config_)
    elif rec_model == "lightgcn":
        rec_config_ = {'user_num':int(user_num), 'item_num':int(item_num), 'embedding_size':64, 'embed_size':64, 'dataset':args.data_ref.strip('/').split('/')[-1]}
        rec_config = Rec_config(rec_config_)
        rec_config.add_para_l(rec_config_)
    elif rec_model == "sasrec":
        rec_config_ = {'user_num':int(user_num), 'item_num':int(item_num), 'hidden_units':64}
        rec_config = Rec_config(rec_config_)
        rec_config.add_para_s(rec_config_)

    rec_encoder = init_rec_encoder(rec_model, rec_config)
    rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location="cpu"))
    if rec_model == "lightgcn":
        from minigpt4.datasets.datasets.rec_gnndataset import GnnDataset
        gnndata = GnnDataset(rec_config, args.data_ref)
        rec_encoder._set_graph(gnndata.Graph)

    for name, param in rec_encoder.named_parameters():
        param.requires_grad = False
    rec_encoder = rec_encoder.eval()
    print("freeze rec encoder")

    users = list(total_.uid)
    # users.extend(list(test_.uid))
    # users.extend(list(valid_.uid))
    users = list(set(users))

    if rec_model == "sasrec":
        total_['len'] = total_['his'].apply(lambda x: len(x))
        total_ = total_.sort_values(['uid', 'len'], ascending=[True, False])
        # max_len_per_user = total_.groupby('uid')['len'].transform('max')
        sas_seqs = []
        for user in users:
            user_records = total_[total_.uid == user]
            longest_record = user_records.iloc[0]
            sas_seq = longest_record['his'].tolist()[-25:]
            if len(sas_seq) < 25:
                pre = [0]*(25-len(sas_seq))
                pre.extend(sas_seq)
                sas_seqs.append(pre)
            else:
                sas_seqs.append(sas_seq)
        user_embeds = rec_encoder.seq_encoder(torch.tensor(sas_seqs)).detach().numpy()
    else:
        user_embeds = rec_encoder.user_encoder(torch.tensor(users)).detach().numpy()
    print('grouped user embedding')
    # 数据标准化 (对基于距离的聚类方法很重要)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(user_embeds)
    # 执行聚类 (这里以K-Means为例)
    # cluster_labels = dbscan_clustering(embeddings_scaled, dim=256)
    if args.mode == "k":
        n_clusters, cluster_labels = kmeans_clustering(embeddings_scaled, n_clusters=n_clusters)
    else:
        cluster_labels = hierarchical_clustering(embeddings_scaled, n_clusters=n_clusters)
    # 将结果保存到DataFrame
    cluster = {}
    if rec_model == "sasrec":
        results = pd.DataFrame({
            'user_id': users,
            'cluster': cluster_labels,
            'embedding': user_embeds.tolist()
        })
    else:
        results = pd.DataFrame({
            'user_id': users,
            'cluster': cluster_labels,
        })
    for user, cluster_label in zip(users,cluster_labels):
        if int(cluster_label) not in cluster:
            cluster[int(cluster_label)] = [user]
        else:
            cluster[int(cluster_label)].append(user)
    save_fig = os.path.join(data_dir,f'{rec_model.lower()}_user_group_{n_clusters}.svg')
    visualize_clusters(embeddings_scaled, cluster_labels, save_fig)
    # 查看聚类结果
    print("聚类结果统计:")
    print(results['cluster'].value_counts())
    # 保存结果到CSV
    for cluster_label,users in cluster.items():
        cluster[cluster_label] = list(set(sorted(users)))
    results.to_csv(os.path.join(data_dir,f'{rec_model.lower()}_user_group_{n_clusters}.csv'), index=False)
    with open(os.path.join(data_dir,f'{rec_model.lower()}_user_group_{n_clusters}.json'),'w')as f:
        json.dump(cluster,f,indent=4)
    print(f'split data into {data_dir}/grouped_{n_clusters}')
    for idx in cluster:
        data_group_dir_idx = os.path.join(data_dir,rec_model.lower(),f'grouped_{n_clusters}',f'group_{idx}')
        os.makedirs(data_group_dir_idx,exist_ok=True)
        os.system(f'ln -s {data_dir}/id2title.json {data_group_dir_idx}/')
        train_idx = train_[train_['uid'].isin(cluster[idx])]
        valid_idx = valid_[valid_['uid'].isin(cluster[idx])]
        # valid_small_idx = valid_small[valid_small['uid'].isin(data_group_dict[idx])]
        # valid_small_idx = valid_idx.sample(frac=0.5,random_state=2025)
        valid_small_idx = valid_idx.sample(n=10000,random_state=2025) if len(valid_idx) > 10000 else valid_idx
        # test_idx = test_[test_['uid'].isin(cluster[idx])]
        reason_idx = reason_[reason_['uid'].isin(cluster[idx])]
        print(f'{idx} data size:{len(train_idx)}, {len(valid_idx)}, {len(reason_idx)}')
        train_idx.to_pickle(os.path.join(data_group_dir_idx,"train_ood2.pkl"))
        valid_idx.to_pickle(os.path.join(data_group_dir_idx,"valid_ood2.pkl"))
        valid_small_idx.to_pickle(os.path.join(data_group_dir_idx,"valid_small_ood2.pkl"))
        # test_idx.to_pickle(os.path.join(data_group_dir_idx,"test_ood2.pkl"))
        reason_idx.to_pickle(os.path.join(data_group_dir_idx,"reason_ood2.pkl"))


