"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
from math import e
import os
import re
from sympy import Determinant, det, im
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue, MetricLogger_auc, SmoothedValue_v2
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
from transformers import GenerationConfig
from sklearn.metrics import roc_auc_score,accuracy_score
from collections import defaultdict
from minigpt4.tasks.base_task import BaseTask
import time
import numpy as np
import math
import evaluate

# def calculate_hr1(eval_content):
#     correct_num=0
#     valid_num=0
#     total_num=0
#     start_time = time.time()
#     for i,generate in enumerate(eval_content["generate"]):
#         real=eval_content["real"][i]
#         cans=eval_content["cans"][i]
#         total_num+=1
#         generate=generate.strip().lower().strip()
#         real=real.strip().lower().strip()
#         cans=[item.strip().lower().strip() for item in cans]
#         gen_cans_list=[]
#         for cans_item in cans:
#             if cans_item in generate:
#                 gen_cans_list.append(cans_item)
#         if len(gen_cans_list)>0:
#             valid_num+=1
#             if real in gen_cans_list:
#                 correct_num+=1
#     valid_ratio=valid_num/total_num
#     if valid_num>0:
#         hr1=correct_num/valid_num
#     else:
#         hr1=0
#     print(f'Calculate HR1 cost: {time.time()-start_time}s...')
#     return valid_ratio,hr1

def calculate_ttr(text):
    tokens = text.split()
    if len(tokens) == 0:
        return 0,0
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens), len(re.findall(r'\b\w+\b', text.lower()))

def calculate_ngram_entropy(text, n=1, smooth=True):
    tokens = re.findall(r'\b\w+\b', text.lower())  # 过滤特殊符号
    if len(tokens) < n:
        return 0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    # 频次统计（添加平滑）
    counts = defaultdict(int)
    for gram in ngrams:
        counts[gram] += 1
    if smooth:
        vocab_size = len(set(tokens))  # 使用词汇表大小作为平滑因子
        total = len(ngrams) + vocab_size
    else:
        total = len(ngrams)
    
    # 熵计算
    entropy = 0.0
    for gram, count in counts.items():
        p = (count + 1)/total if smooth else count/total
        p = max(p, 1e-10)  # 设置最小值
        entropy -= p * math.log(p)
    return entropy / math.log(2)  # 返回bits为单位的熵值


def reorganize_by_user(user, predict, label):
    if not isinstance(predict,np.ndarray):
        predict = np.array(predict)
    if not isinstance(label,np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    discard_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    for u_i in u:
        start_id,end_id = total_num, total_num+counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts ==1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            discard_dict[u_i] = [predict[index_ui], label[index_ui]]
            continue
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        k+=1
    print("only one interaction users:",only_one_interaction)
    return candidates_dict
def uAUC_me(candidates_dict):
    computed_u = []
    auc=[]
    only_one_class = 0

    for ui,pre_and_true in candidates_dict.items():
        pre_i,label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i,pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
        
    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    return uauc, computed_u, auc_for_user

def hr_at_k(candidates_dict, k):
    """
    计算 HR@K（Hit Rate at K）
    :param candidates_dict: {uid: (pred_scores, labels)}
    :param k: Top-K 推荐
    :return: 所有用户的平均 HR@K
    """
    hr_list = []
    for uid,pre_and_true in candidates_dict.items():
        pred,label = pre_and_true
        if len(pred) == 0:
            continue  # 跳过无候选物品的用户
        # 按预测分数降序排序，取 Top-K
        top_k_indices = np.argsort(-np.array(pred))[:k]
        # 检查 Top-K 中是否有至少一个正样本
        hit = np.any(np.array(label)[top_k_indices] == 1)
        hr_list.append(hit)
    return np.mean(hr_list) if hr_list else 0.0

def ndcg_at_k(candidates_dict, k):
    """
    计算 NDCG@K（Normalized Discounted Cumulative Gain at K）
    :param candidates_dict: {uid: (pred_scores, labels)}
    :param k: Top-K 推荐
    :return: 所有用户的平均 NDCG@K
    """
    ndcg_list = []
    for uid,pre_and_true in candidates_dict.items():
        pred,label = pre_and_true
        if len(pred) == 0:
            continue
        # 按预测分数降序排序
        ranked_labels = np.array(label)[np.argsort(-np.array(pred))]
        # 计算 DCG@K
        dcg = 0.0
        for i in range(min(k, len(ranked_labels))):
            rel = ranked_labels[i]
            dcg += rel / np.log2(i + 2)  # log2(i+2) 因为索引从 0 开始
        # 计算 IDCG@K（理想排序的 DCG）
        ideal_labels = np.sort(np.array(label))[::-1]  # 降序排列
        idcg = 0.0
        for i in range(min(k, len(ideal_labels))):
            rel = ideal_labels[i]
            idcg += rel / np.log2(i + 2)
        # 避免除以 0
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_list.append(ndcg)
    return np.mean(ndcg_list) if ndcg_list else 0.0

def map_at_k(candidates_dict, k):
    """
    计算 MAP@K（Mean Average Precision at K）
    :param candidates_dict: {uid: (pred_scores, labels)}
    :param k: Top-K 推荐
    :return: 所有用户的平均 MAP@K
    """
    ap_list = []
    for uid,pre_and_true in candidates_dict.items():
        pred,label = pre_and_true
        if len(pred) == 0:
            continue
        # 按预测分数降序排序
        ranked_labels = np.array(label)[np.argsort(-np.array(pred))]
        # 计算 Precision@K 的累积和
        hits = 0
        sum_precision = 0.0
        for i in range(min(k, len(ranked_labels))):
            if ranked_labels[i] == 1:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precision += precision_at_i
        # 计算 Average Precision (AP)
        ap = sum_precision / np.sum(label) if np.sum(label) > 0 else 0.0
        ap_list.append(ap)
    return np.mean(ap_list) if ap_list else 0.0

# Function to gather tensors across processes
def gather_tensor(tensor, dst=0):
    if dist.is_available():
        world_size = dist.get_world_size()
        if world_size > 1:
            if not isinstance(tensor, list):
                tensor = [tensor]

            gathered_tensors = [torch.empty_like(t) for t in tensor]
            dist.gather(tensor, gathered_tensors, dst=dst)

            return gathered_tensors
        else:
            return tensor
    else:
        return tensor

class RecBaseTask(BaseTask):
    def valid_step(self, model, samples):
        outputs = model.generate_for_samples(samples)
        return outputs
        # raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        pass
        # model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    # def evaluation(self, model, data_loaders, cuda_enabled=True):
    #     model = model.eval()
    #     metric_logger = MetricLogger(delimiter="  ")
    #     auc_logger = MetricLogger(delimiter="  ")
    #     metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     auc_logger.add_meter("auc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     header = "Evaluation"
    #     # TODO make it configurable
    #     print_freq = len(data_loaders.loaders[0])//5 #10

    #     results = []
    #     results_loss = []
    #     results_logits = []
    #     labels = []
    #     k = 0
    #     use_auc = False
    #     for data_loader in data_loaders.loaders:
    #         for samples in metric_logger.log_every(data_loader, print_freq, header):
    #             # samples = next(data_loader)
    #             samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
    #             eval_output = self.valid_step(model=model, samples=samples)
    #             # results_loss.append(eval_output['loss'].item())
    #             if 'logits' in eval_output.keys():
    #                 use_auc = True
    #                 results_logits.extend(eval_output['logits'].detach().cpu().numpy())
    #                 labels.extend(samples['label'].detach().cpu().numpy())
    #                 logits = eval_output['logits']
    #                 logits[logits==0.5] = 1
    #                 acc = (logits-samples['label'])
    #                 acc = (acc==0).sum()/acc.shape[0]
    #                 metric_logger.update(acc=acc.item())
    #             else: 
    #                 metric_logger.update(acc=0)
    #             # acc = accuracy_score(samples['label'].cpu().numpy().astype(int), logits.astype(int))
    #             # results.extend(eval_output)
    #             metric_logger.update(loss=eval_output['loss'].item())
    #             torch.cuda.empty_cache()
            
    #         if use_auc:
    #             auc = roc_auc_score(labels, results_logits)
    #             auc_logger.update(auc=auc)

    #         if is_dist_avail_and_initialized():
    #             dist.barrier()

    #         metric_logger.synchronize_between_processes()
    #         auc_logger.synchronize_between_processes()
    #         auc = 0
    #         # print("Label type......",type(labels),labels)
    #         if use_auc:
    #             auc = roc_auc_score(labels, results_logits)
    #         logging.info("Averaged stats: " + str(metric_logger.global_avg()) + " auc: " + str(auc) + "  global"+ str(auc_logger.global_avg()))
            
    #         if use_auc:
    #             results = {
    #                 'agg_metrics':auc,
    #                 'acc': metric_logger.meters['acc'].global_avg,
    #                 'loss':  metric_logger.meters['loss'].global_avg
    #             }
    #         else: # only loss usable
    #             results = {
    #                 'agg_metrics': -metric_logger.meters['loss'].global_avg,
    #             }

    #     return results

    # def evaluation(self, model, data_loaders, cuda_enabled=True, id2title=None):
    #     model = model.eval()
    #     metric_logger = MetricLogger(delimiter="  ")
    #     auc_logger = MetricLogger(delimiter="  ")
    #     metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     auc_logger.add_meter("auc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     header = "Evaluation"
    #     # TODO make it configurable
    #     print_freq = len(data_loaders.loaders[0])//5 #10

    #     results = []
    #     results_loss = []
        
    #     k = 0
    #     use_auc = False
    #     criterion = nn.BCEWithLogitsLoss()
    #     for data_loader in data_loaders.loaders:
    #         results_logits = []
    #         labels = []
    #         users = []
    #         for samples in metric_logger.log_every(data_loader, print_freq, header):
    #             # samples = next(data_loader)
    #             samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
    #             eval_output = self.valid_step(model=model, samples=samples)
    #             # results_loss.append(eval_output['loss'].item())
    #             if 'logits' in eval_output.keys():
    #                 use_auc = True
    #                 logits = np.array(eval_output['logits'])
    #                 users.extend(samples['UserID'].detach().cpu().numpy())
    #                 results_logits.extend(logits)
    #                 labels.extend(samples['label'].detach().cpu().numpy())
    #                 label = samples['label'].float()
    #                 logits = torch.tensor(logits).to(label.device).float()
    #                 acc = (logits-label)
    #                 acc = (acc==0).sum()/acc.shape[0]
    #                 metric_logger.update(acc=acc.item())
    #             # acc = accuracy_score(samples['label'].cpu().numpy().astype(int), logits.astype(int))
    #             # results.extend(eval_output)
    #             metric_logger.update(loss=criterion(logits, label).item())
    #             torch.cuda.empty_cache()
    #         # results_logits_ = torch.tensor(results_logits).to(eval_output['logits'].device).contiguous()
    #         # labels_ = torch.tensor(labels).to(eval_output['logits'].device).contiguous()
    #         # users_ = torch.tensor(users).to(eval_output['logits'].device).contiguous()
    #         results_logits_ = np.array(results_logits)
    #         labels_ = np.array(labels)
    #         users_ = np.array(users)
    #         # if use_auc:
    #         #     labels = dist.gather_object()
    #         #     auc = roc_auc_score(labels, results_logits)
    #         #     auc_logger.update(auc=auc)
    #         auc = 0
    #         auc = roc_auc_score(labels_.cpu().numpy(), results_logits_.cpu().numpy())
            # user_dict = reorganize_by_user(users_.cpu().numpy(), results_logits_.cpu().numpy(), labels_.cpu().numpy())
            # uauc = uAUC_me(user_dict)
            
            
    #         metric_logger.synchronize_between_processes()
    #         # auc_logger.synchronize_between_processes()
    #         # auc = 0
    #         # # print("Label type......",type(labels),labels)
    #         if use_auc:
    #             auc_rank0 = roc_auc_score(labels_.cpu().numpy(), results_logits_.cpu().numpy())
    #         logging.info("Averaged stats: " + str(metric_logger.global_avg()) + " ***auc: " + str(auc) + " ***uauc:" +str(uauc) )
    #         print("rank_0 auc:", str(auc_rank0))
            
    #         if use_auc:
    #             results = {
    #                 'agg_metrics':auc,
    #                 'acc': metric_logger.meters['acc'].global_avg,
    #                 'loss':  metric_logger.meters['loss'].global_avg,
    #                 'uauc': uauc
    #             }
    #         else: # only loss usable
    #             results = {
    #                 'agg_metrics': -metric_logger.meters['loss'].global_avg,
    #             }

    #     return results

    # def evaluation(self, model, data_loaders, cuda_enabled=True, id2title=None):#recall
    #     model = model.eval()
    #     metric_logger = MetricLogger(delimiter="  ")
    #     auc_logger = MetricLogger(delimiter="  ")
    #     # metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     metric_logger.add_meter("hr1", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     metric_logger.add_meter("valid_ratio", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     # metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     # auc_logger.add_meter("auc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     header = "Evaluation"
    #     # TODO make it configurable
    #     print_freq = len(data_loaders.loaders[0])//5 #10
    #     print(f'print_freq: {print_freq}')

    #     results = []
        
    #     k = 0
    #     use_auc = False
    #     for data_loader in data_loaders.loaders:
    #         eval_content = {}
    #         users = []
    #         for samples in metric_logger.log_every(data_loader, print_freq, header):
    #             # samples = next(data_loader)
    #             samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
    #             eval_output = self.valid_step(model=model, samples=samples)
    #             eval_content['generate'] = eval_output
    #             bsz = samples['UserID'].shape[0]
    #             TargetItemID = samples['TargetItemID'].detach().cpu().numpy()
    #             CandidateItemIDs = samples['CandidateItemIDs'].detach().cpu().numpy()
    #             eval_content['real'] = [id2title[str(TargetItemID[row])]['title'] for row in range(bsz)]
    #             eval_content['cans'] = [[id2title[str(idx)]['title'] for idx in CandidateItemIDs[row]] for row in range(bsz)]
    #             # print('eval_content',eval_content)
    #             users.extend(samples['UserID'].detach().cpu().numpy())
    #             valid_ratio,hr1 = calculate_hr1(eval_content)

    #             metric_logger.update(hr1=hr1)
    #             metric_logger.update(valid_ratio=valid_ratio)

    #             # torch.cuda.empty_cache()
    #         users_ = torch.tensor(users).to(eval_output.device).contiguous()
            
    #         metric_logger.synchronize_between_processes()
    #         logging.info("Averaged stats: " + str(metric_logger.global_avg()))
            
    #         results = {
    #             'agg_metrics':hr1,
    #             'hr1': metric_logger.meters['hr1'].global_avg,
    #             'valid_ratio':  metric_logger.meters['valid_ratio'].global_avg,
    #         }

    #     return results


    def evaluation(self, model, data_loaders, cuda_enabled=True, eval_text=False):
        model = model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        # auc_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("PPL", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("TTR", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("length", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("ngram2", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("ngram3", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("ngram4", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        valid_ratio = {"TTR":0,"ngram2":0,"ngram3":0,"ngram4":0}
        deterministic = False
        total_valid = 0
        header = "Evaluation"
        # TODO make it configurable
        print_freq = len(data_loaders.loaders[0])//5 #10

        results = []
        results_loss = []
        perplexity = None
        
        k = 0
        use_auc = False
        eval_times = []
        metric_logger.update(TTR=-1e-7)
        metric_logger.update(PPL=-1e-7)
        metric_logger.update(ngram2=-1e-7)
        metric_logger.update(ngram3=-1e-7)
        metric_logger.update(ngram4=-1e-7)
        for data_loader in data_loaders.loaders:
            results_logits = []
            labels = []
            users = []
            for samples in metric_logger.log_every(data_loader, print_freq, header):
                # samples = next(data_loader)
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                # epoch_start = time.time()
                eval_output = self.valid_step(model=model, samples=samples)
                # epoch_end = time.time()
                eval_times.append(eval_output['cost'])
                # results_loss.append(eval_output['loss'].item())
                if 'logits' in eval_output.keys():
                    use_auc = True
                    users.extend(samples['UserID'].detach().cpu().numpy())
                    results_logits.extend(eval_output['logits'].detach().cpu().numpy())
                    labels.extend(samples['label'].detach().cpu().numpy())
                    logits = eval_output['logits']
                    logits[logits>0.5] = 1
                    acc = (logits-samples['label'])
                    acc = (acc==0).sum()/acc.shape[0]
                    metric_logger.update(acc=acc.item())
                else:
                    metric_logger.update(acc=0)

                if 'reason_text' in eval_output.keys() and eval_text:
                    if perplexity is None:
                        perplexity = evaluate.load("minigpt4/tasks/perplexity.py", module_type="metric")
                    reason_text = eval_output['reason_text']
                    total_valid += samples['label'].shape[0]
                    #PPL
                    try:
                        results = perplexity.compute(model_id=(model.llama_model_lora, model.llama_tokenizer),add_start_token=False,predictions=reason_text,max_length=1024)
                        metric_logger.update(PPL=results["mean_perplexity"])
                    except:
                        print('perplexity error')
                    #TTR, entropy_ngram
                    for text in reason_text:
                        ttr_score,length = calculate_ttr(text)
                        entropy_2gram= calculate_ngram_entropy(text,2)
                        entropy_3gram= calculate_ngram_entropy(text,3)
                        entropy_4gram= calculate_ngram_entropy(text,4)
                        metric_logger.update(length=length)
                        if ttr_score != 0:
                            valid_ratio['TTR'] += 1
                            metric_logger.update(TTR=ttr_score)
                        if entropy_2gram != 0:
                            valid_ratio['ngram2'] += 1
                            metric_logger.update(ngram2=entropy_2gram)
                        if entropy_3gram != 0:
                            valid_ratio['ngram3'] += 1
                            metric_logger.update(ngram3=entropy_3gram)
                        if entropy_4gram != 0:
                            valid_ratio['ngram4'] += 1
                            metric_logger.update(ngram4=entropy_4gram)
                else:
                    if not deterministic:
                        metric_logger.del_meter("PPL")
                        metric_logger.del_meter("length")
                        metric_logger.del_meter("TTR")
                        metric_logger.del_meter("ngram2")
                        metric_logger.del_meter("ngram3")
                        metric_logger.del_meter("ngram4")
                        deterministic = True
                metric_logger.update(loss=eval_output['loss'].item())
                torch.cuda.empty_cache()
            results_logits_ = torch.tensor(results_logits).to(eval_output['logits'].device).contiguous()
            labels_ = torch.tensor(labels).to(eval_output['logits'].device).contiguous()
            users_ = torch.tensor(users).to(eval_output['logits'].device).contiguous()
            # if use_auc:
            #     labels = dist.gather_object()
            #     auc = roc_auc_score(labels, results_logits)
            #     auc_logger.update(auc=auc)
            auc = 0
            if is_dist_avail_and_initialized():
                print("wating comput auc.....")
                rank = dist.get_rank()
                gathered_labels = [labels_.clone() for _ in range(dist.get_world_size())]
                gathered_logits = [results_logits_.clone() for _ in range(dist.get_world_size())]
                gathered_users = [users_.clone() for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_labels, labels_)
                dist.all_gather(gathered_logits, results_logits_)
                dist.all_gather(gathered_users, users_)
                
                labels_a = torch.cat(gathered_labels,dim=0).flatten().cpu().numpy()
                results_logits_a = torch.cat(gathered_logits,dim=0).flatten().cpu().numpy()
                users_a = torch.cat(gathered_users,dim=0).flatten().cpu().numpy()
                print("computing....")
                auc = roc_auc_score(labels_a, results_logits_a)
                user_dict = reorganize_by_user(users_a,results_logits_a,labels_a)
                uauc, _, _ = uAUC_me(user_dict)
                print("finished comput auc.....")
            else:
                auc = roc_auc_score(labels_.cpu().numpy(), results_logits_.cpu().numpy())
                user_dict = reorganize_by_user(users_.cpu().numpy(), results_logits_.cpu().numpy(), labels_.cpu().numpy())
                uauc = uAUC_me(user_dict)
            
            if is_dist_avail_and_initialized():
                dist.barrier()
                # dist.reduce()
            
            metric_logger.synchronize_between_processes()
            # auc_logger.synchronize_between_processes()
            # auc = 0
            # # print("Label type......",type(labels),labels)
            # if use_auc:
            #     auc_rank0 = auc#roc_auc_score(labels_.cpu().numpy(), results_logits_.cpu().numpy())
            # logging.info("Averaged stats: " + str(metric_logger.global_avg()) + " ***auc: " + str(auc) + " ***uauc:" +str(uauc[0]) )
            logging.info(f"Averaged stats: {metric_logger.global_avg()} ***auc: {auc:.4f} ***uauc:{uauc[0]:.4f}")
            # print("rank_0 auc:", str(auc_rank0))
            if 'reason_text' in eval_output.keys() and eval_text:#eval_only
                for metrick,metricv in valid_ratio.items():
                    print(f'{metrick}_valid_ratio: {metricv/total_valid}')

            for topk in [1, 2, 3, 5, 10]:
                print(f"HR@{topk}: {hr_at_k(user_dict, topk):.4f}, NDCG@{topk}: {ndcg_at_k(user_dict, topk):.4f}, MAP@{topk}: {map_at_k(user_dict, topk):.4f}")

            if use_auc:
                results = {
                    'agg_metrics':auc,
                    'acc': metric_logger.meters['acc'].global_avg,
                    'loss':  metric_logger.meters['loss'].global_avg,
                    'uauc': uauc
                }
            else: # only loss usable
                results = {
                    'agg_metrics': -metric_logger.meters['loss'].global_avg,
                }
        for_time_cal_start_idx = len(eval_times)//4
        for_time_cal = eval_times[for_time_cal_start_idx:for_time_cal_start_idx*3]
        print(f"eval_times:(inner {len(for_time_cal)} iters)", sum(for_time_cal)/len(for_time_cal))
        return results