"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue, MetricLogger_auc, SmoothedValue_v2
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
from transformers import GenerationConfig
from sklearn.metrics import roc_auc_score,accuracy_score
from minigpt4.tasks.base_task import BaseTask
import time
import numpy as np

def calculate_hr1(eval_content):
    correct_num=0
    valid_num=0
    total_num=0
    start_time = time.time()
    for i,generate in enumerate(eval_content["generate"]):
        real=eval_content["real"][i]
        cans=eval_content["cans"][i]
        total_num+=1
        generate=generate.strip().lower().strip()
        real=real.strip().lower().strip()
        cans=[item.strip().lower().strip() for item in cans]
        gen_cans_list=[]
        for cans_item in cans:
            if cans_item in generate:
                gen_cans_list.append(cans_item)
        if len(gen_cans_list)>0:
            valid_num+=1
            if real in gen_cans_list:
                correct_num+=1
    valid_ratio=valid_num/total_num
    if valid_num>0:
        hr1=correct_num/valid_num
    else:
        hr1=0
    print(f'Calculate HR1 cost: {time.time()-start_time}s...')
    return valid_ratio,hr1


def uAUC_me(user, predict, label):
    predict = predict.squeeze()
    label = label.squeeze()
    start_time = time.time()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    discard_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
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
            # print("only one class")
        
    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time()-start_time,'uauc:', uauc)
    return uauc, computed_u, auc_for_user

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
    #         uauc = uAUC_me(users_.cpu().numpy(), results_logits_.cpu().numpy(), labels_.cpu().numpy())
            
            
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


    def evaluation(self, model, data_loaders, cuda_enabled=True):
        model = model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        auc_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        auc_logger.add_meter("auc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Evaluation"
        # TODO make it configurable
        print_freq = len(data_loaders.loaders[0])//5 #10

        results = []
        results_loss = []
        
        k = 0
        use_auc = False
        eval_times = []
        for data_loader in data_loaders.loaders:
            results_logits = []
            labels = []
            users = []
            for samples in metric_logger.log_every(data_loader, print_freq, header):
                # samples = next(data_loader)
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                epoch_start = time.time()
                eval_output = self.valid_step(model=model, samples=samples)
                epoch_end = time.time()
                eval_times.append(epoch_end-epoch_start)
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
                # acc = accuracy_score(samples['label'].cpu().numpy().astype(int), logits.astype(int))
                # results.extend(eval_output)
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
                uauc, _, _ = uAUC_me(users_a,results_logits_a,labels_a)
                print("finished comput auc.....")
            else:
                auc = roc_auc_score(labels_.cpu().numpy(), results_logits_.cpu().numpy())
                uauc = uAUC_me(users_.cpu().numpy(), results_logits_.cpu().numpy(), labels_.cpu().numpy())
            

            if is_dist_avail_and_initialized():
                dist.barrier()
                # dist.reduce()
            
            metric_logger.synchronize_between_processes()
            # auc_logger.synchronize_between_processes()
            # auc = 0
            # # print("Label type......",type(labels),labels)
            if use_auc:
                auc_rank0 = auc#roc_auc_score(labels_.cpu().numpy(), results_logits_.cpu().numpy())
            logging.info("Averaged stats: " + str(metric_logger.global_avg()) + " ***auc: " + str(auc) + " ***uauc:" +str(uauc[0]) )
            print("rank_0 auc:", str(auc_rank0))
            
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
        print("eval_times:(inner 100 iters)", sum(eval_times[50:150])/100)
        return results