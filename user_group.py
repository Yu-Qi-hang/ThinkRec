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

class Rec_config:
    def __init__(self, user_num, item_num, embedding_size):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = embedding_size

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
def kmeans_clustering(data, max_clusters=10):
    silhouette_scores = []
    cluster_range = range(2, max_clusters+1, 2)
    
    for n_clusters in tqdm(cluster_range):
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"最佳聚类数: {best_n_clusters}")
    final_kmeans = MiniBatchKMeans(n_clusters=best_n_clusters, random_state=42, batch_size=1024, n_init=10)
    return final_kmeans.fit_predict(data)

# 方法2: 层次聚类
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


# 可视化 (使用t-SNE降维到2D)
def visualize_clusters(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of User Clusters')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('user_clusters.png')




data_dir = "/home/yuqihang/projects/CoLLM/collm-datasets/bookdu"
train_ = pd.read_pickle(os.path.join(data_dir,"train_ood2.pkl"))
valid_ = pd.read_pickle(os.path.join(data_dir,"valid_ood2.pkl"))
test_ = pd.read_pickle(os.path.join(data_dir,"test_ood2.pkl"))
user_num = max(train_.uid.max(),valid_.uid.max(),test_.uid.max())+1
item_num = max(train_.iid.max(),valid_.iid.max(),test_.iid.max())+1

print('Loading Rec_model')
rec_model = "MF"
rec_config = Rec_config(user_num=int(user_num), item_num=int(item_num), embedding_size=256)
rec_encoder = init_rec_encoder(rec_model, rec_config)

pretrained_rec = "/data/yuqihang/result/CoLLM/checkpoints/mf/0228_booknew_best_model_d256_lr0.001_wd1e-06.pth"
rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location="cpu"))
print("successfully load the pretrained model......")
for name, param in rec_encoder.named_parameters():
    param.requires_grad = False
rec_encoder = rec_encoder.eval()
# rec_encoder.train = disabled_train
print("freeze rec encoder")
print('Loading Rec_model Done')

users = list(train_.uid)
users.extend(list(test_.uid))
users.extend(list(valid_.uid))
users = list(set(users))

user_embeds = rec_encoder.user_encoder(torch.tensor(users)).detach().numpy()

# 数据标准化 (对基于距离的聚类方法很重要)
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(user_embeds)
# 执行聚类 (这里以K-Means为例)
cluster_labels = kmeans_clustering(embeddings_scaled, max_clusters=5)
# cluster_labels = dbscan_clustering(embeddings_scaled, dim=256)
# cluster_labels = hierarchical_clustering(embeddings_scaled, n_clusters=8)
# 将结果保存到DataFrame
cluster = {}
results = pd.DataFrame({
    'user_id': users,
    'cluster': cluster_labels
})
for user, cluster_label in zip(users,cluster_labels):
    if int(cluster_label) not in cluster:
        cluster[int(cluster_label)] = [user]
    else:
        cluster[int(cluster_label)].append(user)
visualize_clusters(embeddings_scaled, cluster_labels)
# 查看聚类结果
print("聚类结果统计:")
print(results['cluster'].value_counts())
# 保存结果到CSV
for cluster_label,users in cluster.items():
    cluster[cluster_label] = list(set(sorted(users)))
results.to_csv(os.path.join(data_dir,'user_group.csv'), index=False)
with open(os.path.join(data_dir,'user_group.json'),'w')as f:
    json.dump(cluster,f,indent=4)

# for idx in cluster:
#     data_group_dir_idx = os.path.join(data_dir,f'group_{idx}')
#     os.makedirs(data_group_dir_idx,exist_ok=True)
#     train_idx = train_[train_['uid'].isin(data_group_dict[idx])]
#     valid_idx = valid_[valid_['uid'].isin(data_group_dict[idx])]
#     # valid_small_idx = valid_small[valid_small['uid'].isin(data_group_dict[idx])]
#     valid_small_idx = valid_idx.sample(frac=0.5,random_state=2025)
#     test_idx = test_[test_['uid'].isin(data_group_dict[idx])]
#     reason_idx = reason_[reason_['uid'].isin(data_group_dict[idx])]
#     train_idx.to_pickle(os.path.join(data_group_dir_idx,"train_ood2.pkl"))
#     valid_idx.to_pickle(os.path.join(data_group_dir_idx,"valid_ood2.pkl"))
#     valid_small_idx.to_pickle(os.path.join(data_group_dir_idx,"valid_small_ood2.pkl"))
#     test_idx.to_pickle(os.path.join(data_group_dir_idx,"test_ood2.pkl"))
#     reason_idx.to_pickle(os.path.join(data_group_dir_idx,"reason_ood2.pkl"))


