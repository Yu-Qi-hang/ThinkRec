from minigpt4.models.rec_model import MatrixFactorization
from minigpt4.tasks.rec_base_task import reorganize_by_user, uAUC_me, hr_at_k, map_at_k, ndcg_at_k

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import argparse
import random 
import datetime
import time
import os


class early_stoper(object):
    def __init__(self,ref_metric='valid_auc', incerase =True,patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = incerase
        self.reach_count = 0
        self.patience= patience
        # self.metrics = None
    
    def _registry(self,metrics):
        self.best_metric = metrics

    def update(self, metrics):
        if self.best_metric is None:
            self._registry(metrics)
            return True
        else:
            if self.increase and metrics[self.ref_metric] > self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            elif not self.increase and metrics[self.ref_metric] < self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True 
            else:
                self.reach_count += 1
                return False

    def is_stop(self):
        if self.reach_count>=self.patience:
            return True
        else:
            return False

# set random seed   
def run_a_trail(train_config, data_dir='', log_file=None, save_mode=False, save_file=None, need_train=True, warm_or_cold=None):
    seed=2025
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load dataset
    train_data = pd.read_pickle(os.path.join(data_dir,"train_ood2.pkl"))[['uid','iid', 'label']].values
    valid_data = pd.read_pickle(os.path.join(data_dir,"valid_ood2.pkl"))[['uid','iid', 'label']].values
    test_data = pd.read_pickle(os.path.join(data_dir,"test_ood2.pkl"))[['uid','iid', 'label']].values


    user_num = max(train_data[:,0].max(), valid_data[:,0].max(), test_data[:,0].max()) + 1
    item_num =  max(train_data[:,1].max(), valid_data[:,1].max(), test_data[:,1].max()) + 1

    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(os.path.join(data_dir,"test_warm_cold_ood2.pkl"))[['uid','iid','label', 'warm']]
            test_data = test_data[test_data['warm'].isin([1])][['uid','iid','label']].values
            print("warm data size:", test_data.shape[0])
            # pass
        else:
            test_data = pd.read_pickle(os.path.join(data_dir,"test_warm_cold_ood2.pkl"))[['uid','iid','label', 'cold']]
            test_data = test_data[test_data['cold'].isin([1])][['uid','iid','label']].values
            print("cold data size:", test_data.shape[0])
            # pass

    print("user nums:", user_num, "item nums:", item_num)

    mf_config={
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size'])
        }
    mf_config = omegaconf.OmegaConf.create(mf_config)

    train_data_loader = DataLoader(train_data, batch_size = train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size = train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size = train_config['batch_size'], shuffle=False)

    model = MatrixFactorization(mf_config).cuda()
    opt = torch.optim.Adam(model.parameters(),lr=train_config['lr'],weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc',incerase=True,patience=train_config['patience'])
    # trainig part
    criterion = nn.BCEWithLogitsLoss()

    if not need_train:
        test_small_path = os.path.join(data_dir,"test_small_ood2.pkl")
        if os.path.exists(test_small_path):
            test_data = pd.read_pickle(test_small_path)[['uid','iid', 'label']].values
            test_data_loader = DataLoader(test_data, batch_size = train_config['batch_size'], shuffle=False)
        model.load_state_dict(torch.load(save_file))
        model.eval()
        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(test_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
            users.extend(batch_data[:,0].cpu().numpy())
        test_auc = roc_auc_score(label,pre)
        test_logloss = F.binary_cross_entropy_with_logits(torch.tensor(pre), torch.tensor(label).float()).item()  # 计算logloss
        user_dict = reorganize_by_user(users, pre, label)
        test_uauc,_,_ = uAUC_me(user_dict)
        label = np.array(label)
        pre = np.array(pre)
        thre = 0.1
        pre[pre>=thre] =  1
        pre[pre<thre]  =0
        test_acc = (label==pre).mean()
        print(f"test auc:{test_auc:.4f}, uauc:{test_uauc:.4f}, acc: {test_acc:.4f}, logloss: {test_logloss:.4f}")
        for topk in [1, 2, 3, 5, 10]:
            print(f"HR@{topk}: {hr_at_k(user_dict, topk):.4f}, NDCG@{topk}: {ndcg_at_k(user_dict, topk):.4f}, MAP@{topk}: {map_at_k(user_dict, topk):.4f}")
        return 
    

    for epoch in range(train_config['epoch']):
        model.train()
        for bacth_id, batch_data in enumerate(train_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            loss = criterion(ui_matching,batch_data[:,-1].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        if epoch% train_config['eval_epoch']==0:
            model.eval()
            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(valid_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
                users.extend(batch_data[:,0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
            valid_auc = roc_auc_score(label,pre)
            user_dict = reorganize_by_user(users, pre, label)
            valid_uauc,_,_ = uAUC_me(user_dict)
            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(test_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
                users.extend(batch_data[:,0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
            test_auc = roc_auc_score(label,pre)
            user_dict = reorganize_by_user(users, pre, label)
            test_uauc,_,_ = uAUC_me(user_dict)
            updated = early_stop.update({'valid_auc':valid_auc, 'valid_uauc':valid_uauc,'test_auc':test_auc, 'test_uauc':test_uauc, 'epoch':epoch})
            if updated and save_mode:
                torch.save(model.state_dict(),save_file)


            print("epoch:{}, valid_auc:{}, test_auc:{}, early_count:{}".format(epoch, valid_auc, test_auc, early_stop.reach_count))
            if early_stop.is_stop():
                print("early stop is reached....!")
                break
            if epoch>500 and early_stop.best_metric[early_stop.ref_metric] < 0.52:
                print("training reaches to 500 epoch but the valid_auc is still less than 0.55")
                break
    print("train_config:", train_config,"\nbest result:",early_stop.best_metric) 
    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)
        log_file.flush()


#### /data/zyang/LLM/PretrainedModels/mf/best_model_d128.pth
# with prtrain version:
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/yuqihang/projects/CoLLM/collm-datasets/booknew")
    parser.add_argument("--save_path", type=str, default="/data/yuqihang/result/CoLLM/checkpoints/mf/")
    parser.add_argument("--istrain", action='store_true')
    args = parser.parse_args()
    istrain = args.istrain

    if istrain:
        lr_ = [1e-4]
        wd_ = [1e-5,1e-6,1e-7]
        embedding_size_ = [256]
        data_dir = args.data_dir
        save_path = args.save_path
        os.makedirs(save_path, exist_ok=True)
        print(f"data_dir:{data_dir}, save_path:{save_path}, istrain:{istrain}")

        logfile = f"{data_dir.strip('/').split('/')[-1].replace('-','')}_best_model.txt"
        f=open(os.path.join(save_path,logfile),'a')

        for lr in lr_:
            for wd in wd_:
                for embedding_size in embedding_size_:
                    model_name = f"{datetime.datetime.now().strftime('%m%d')}_{data_dir.strip('/').split('/')[-1].replace('-','')}_best_model_d{embedding_size}_lr{lr}_wd{wd}.pth"
                    train_config={
                        'lr': lr,
                        'wd': wd,
                        'embedding_size': embedding_size,
                        "epoch": 5000,
                        "eval_epoch":1,
                        "patience":50,
                        "batch_size":1024
                    }
                    print(train_config)
                    run_a_trail(train_config=train_config, data_dir=data_dir, log_file=f, save_mode=istrain, save_file=os.path.join(save_path, model_name), need_train=istrain)
        if f is not None:
            f.close()
    else:
        data_dir = args.data_dir
        model_path = args.save_path
        settings = model_path.replace('.pth','').split('_')
        embedding_size, lr, wd  = int(settings[-3][1:]), float(settings[-2][2:]), float(settings[-1][2:])
        print(f"data_dir:{data_dir}, model_path:{model_path}, istrain:{istrain}")
        train_config={
            'lr': lr,
            'wd': wd,
            'embedding_size': embedding_size,
            "epoch": 5000,
            "eval_epoch":1,
            "patience":50,
            "batch_size":1024
        }
        print(train_config)
        run_a_trail(train_config=train_config, data_dir=data_dir, log_file=None, save_mode=istrain, save_file=model_path, need_train=istrain)