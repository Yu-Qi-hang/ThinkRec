import os
from select import select
# from PIL import Image
# import webdataset as wds
from minigpt4.datasets.datasets.rec_base_dataset import RecBaseDataset
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import logging
import json
import re
import random

def turn2list(sample):
    if isinstance(sample,list):
        sample = sample
    elif isinstance(sample, str):
        sample = [sample]
    else:
        try:
            sample = list(sample)
        except:
            raise ValueError("sample must be a list or a str")
    return sample
def convert_title_list_v3(samples=None,withid=False):
    if len(samples) == 2:
        titles,descs = samples
    elif len(samples) == 1:
        titles = samples
    else:
        raise ValueError("size of samples must be 1 or 2")

    titles = turn2list(titles)
    itemsymbol = '<ItemID>'
    items = []

    if descs is not None:
        descs = turn2list(descs)
        for title,desc in zip(titles,descs):
            title = title.strip(' ')
            desc = desc.strip(' ')
            desc = re.sub(r'<[^>]+>', '', desc)
            desc = re.sub(r'\s+', ' ', desc)
            if withid:
                items.append(f'"{title}" with feature {itemsymbol}  and description: {desc}')
            else:
                items.append(f'"{title}" with description: {desc}')
    else:
        for title in titles:
            title = title.strip(' ')
            if withid:
                items.append(f'"{title}" with feature {itemsymbol}')
            else:
                items.append(f'"{title}"')
    # print("Example of prompts:", items[0])

def convert_title_list_v4(samples=None,labels=None,withid=False):
    if len(samples) == 2:
        titles,descs = samples
    elif len(samples) == 1:
        titles = samples
        desc = None
    else:
        raise ValueError("size of samples must be 1 or 2")

    titles = turn2list(titles)
    if labels is not None:
        label2eval = {0:"No",1:"Yes"}
        labels = [label2eval[label] for label in turn2list(labels)]
    if descs is not None:
        descs = turn2list(descs)
    
    itemsymbol = '<ItemID>'
    items = []

    for idx, title in enumerate(titles):
        if title.strip(" ")=="":
            item = f'"Unknow"'
        else:
            item = f'"{title.strip(" ")}"'
        if withid:
            item = f'{item} with feature {itemsymbol}'
        if labels is not None:
            item = f'{item} (evaluation: {labels[idx]})'
        if descs is not None:
            desc = descs[idx].strip(' ')
            desc = re.sub(r'<[^>]+>', '', desc)
            desc = re.sub(r'\s+', ' ', desc)
            item = f'{item} with description: {desc}'
        items.append(item)
    # print("Example of prompts:", items[0])

    if len(items)>0:
        return "; ".join(items)
    else:
        return "unkow"

def convert_title_list_v2(titles):
    titles_ = []
    for x in titles:
        if len(x)>0:
            titles_.append("\""+ x + "\"")
    if len(titles_)>0:
        return ", ".join(titles_)
    else:
        return "unkow"
def convert_title_list(titles):
    titles = ["\""+ x + "\"" for x in titles]
    return ", ".join(titles)


# class MoiveOOData(RecBaseDataset):
#     def __init__(self, text_processor=None, ann_paths=None, seq_len=None):
#         super().__init__()
#         # self.vis_root = vis_root
#         # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
#         ann_paths = ann_paths[0].split("=")
#         self.annotation = pd.read_pickle(ann_paths[0]+"_ood2.pkl").reset_index(drop=True)

#         ## warm test:
#         if "warm" in ann_paths:
#             self.annotation = self.annotation[self.annotation['warm'].isin([1])].copy()
#         if "cold" in ann_paths:
#             self.annotation = self.annotation[self.annotation['not_cold'].isin([0])].copy()

#         self.use_his = False
#         self.prompt_flag = False

#         if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
#             used_columns = ['uid','iid','title','his', 'his_title','label']
#             renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
#             if 'not_cold' in self.annotation.columns:
#                 used_columns.append("not_cold")
#                 renamed_columns.append("prompt_flag")
#                 self.prompt_flag = True

#             self.use_his = True
#             self.annotation = self.annotation[used_columns]
#             self.annotation.columns = renamed_columns
        
#             self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
#             self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
#             # self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"] #.map(convert_title_list)
#         else:
#             used_columns = ['uid','iid','title','label']
#             renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']
#             if 'not_cold' in self.annotation.columns:
#                 used_columns.append('not_cold')
#                 renamed_columns.append("prompt_flag")
#                 self.prompt_flag = True
#             self.annotation = self.annotation[used_columns]
#             self.annotation.columns = renamed_columns
        
#         print("data path:", ann_paths[0], "data size:", self.annotation.shape)
#         self.user_num = self.annotation['UserID'].max()+1
#         self.item_num = self.annotation['TargetItemID'].max()+1
#         self.text_processor = text_processor
        
        
#         if self.use_his:
#             max_length_ = 0
#             for x in self.annotation['InteractedItemIDs'].values:
#                 max_length_ = max(max_length_, len(x))
#             self.max_lenght = min(max_length_, seq_len) # average: only 50; 0915: 15 
#             print("Movie OOD datasets, max history length:", self.max_lenght)
#             logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))
            
#     def __getitem__(self, index):

#         # TODO this assumes image input, not general enough
#         ann = self.annotation.iloc[index]
#         if self.use_his:
#             a = ann['InteractedItemIDs']
#             InteractedNum = len(a)
#             if a[0] == 0:
#                 InteractedNum -= 1

#             if len(a) < self.max_lenght:
#                 b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
#                 b.extend(a)
#             elif len(a)> self.max_lenght:
#                 b = a[-self.max_lenght:]
#                 InteractedNum = self.max_lenght
#             else:
#                 b = a
#             one_sample = {
#                 "UserID": ann['UserID'],
#                 "InteractedItemIDs_pad": np.array(b),
#                 "InteractedItemIDs": ann['InteractedItemIDs'],
#                 "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
#                 "TargetItemID": ann["TargetItemID"],
#                 "TargetItemTitle": "\""+ann["TargetItemTitle"].strip(' ')+"\"",
#                 "InteractedNum": InteractedNum,
#                 "label": ann['label']
#             }
#             if self.prompt_flag:
#                 one_sample['prompt_flag'] = ann['prompt_flag']
#             return one_sample 
#         else:
#             one_sample = {
#                 "UserID": ann['UserID'],
#                 # "InteractedItemIDs_pad": None,
#                 # # "InteractedItemIDs": ann['InteractedItemIDs'],
#                 # "InteractedItemTitles": None,
#                 "TargetItemID": ann["TargetItemID"],
#                 "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
#                 # "InteractedNum": None,
#                 "label": ann['label']
#             }
#             if self.prompt_flag:
#                 one_sample['prompt_flag'] = ann['prompt_flag']
#             return one_sample 


# class MoiveOOData_sasrec(RecBaseDataset):
#     def __init__(self, text_processor=None, ann_paths=None, seq_len=None, sas_seq_len=25):
#         super().__init__()
#         # self.vis_root = vis_root
#         # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
#         ann_paths = ann_paths[0].split("=")
#         self.annotation = pd.read_pickle(ann_paths[0]+"_ood2.pkl").reset_index(drop=True)
#         # self.annotation = pd.read_pickle(ann_paths[0]+"_ood2.pkl").reset_index(drop=True)
        
#         self.use_his = False
#         self.prompt_flag = False
#         self.sas_seq_len = sas_seq_len

#         if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
#             used_columns = ['uid','iid','title','his', 'his_title','label']
#             renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
#             if 'not_cold' in self.annotation.columns:
#                 used_columns.append("not_cold")
#                 renamed_columns.append("prompt_flag")
#                 self.prompt_flag = True

#             self.use_his = True
#             self.annotation = self.annotation[used_columns]
#             self.annotation.columns = renamed_columns
        
#             self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
#             self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
#             self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"] #.map(convert_title_list)
#         else:
#             used_columns = ['uid','iid','title','label']
#             renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']
#             if 'not_cold' in self.annotation.columns:
#                 used_columns.append('not_cold')
#                 renamed_columns.append("prompt_flag")
#                 self.prompt_flag = True
#             self.annotation = self.annotation[used_columns]
#             self.annotation.columns = renamed_columns
        
#         print("data path:", ann_paths[0], "data size:", self.annotation.shape)
#         self.user_num = self.annotation['UserID'].max()+1
#         self.item_num = self.annotation['TargetItemID'].max()+1
#         self.text_processor = text_processor
        
        
#         if self.use_his:
#             max_length_ = 0
#             for x in self.annotation['InteractedItemIDs'].values:
#                 max_length_ = max(max_length_, len(x))
#             self.max_lenght = min(max_length_, seq_len) # average: only 50; 0915: 15 
#             print("Movie OOD datasets, max history length:", self.max_lenght)
#             logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))
            
#     def __getitem__(self, index):

#         # TODO this assumes image input, not general enough
#         ann = self.annotation.iloc[index]
#         if self.use_his:
#             a = ann['InteractedItemIDs']
#             InteractedNum = len(a)
#             if a[0] == 0:
#                 InteractedNum -= 1

#             if len(a) < self.max_lenght:
#                 b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
#                 b.extend(a)
#             elif len(a)> self.max_lenght:
#                 b = a[-self.max_lenght:]
#                 InteractedNum = self.max_lenght
#             else:
#                 b = a
            
#             if len(a) < self.sas_seq_len: # used for sasrec
#                 c = [0]*(self.sas_seq_len - len(a))
#                 c.extend(a)
#             elif len(a) >= self.sas_seq_len:
#                 c = a[-self.sas_seq_len:]

#             one_sample = {
#                 "UserID": ann['UserID'],
#                 "InteractedItemIDs_pad": np.array(b),
#                 "InteractedItemIDs": ann['InteractedItemIDs'],
#                 "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
#                 "TargetItemID": ann["TargetItemID"],
#                 "TargetItemTitle": "\""+ann["TargetItemTitle"].strip(' ')+"\"",
#                 "InteractedNum": InteractedNum,
#                 "label": ann['label'],
#                 "sas_seq": np.array(c)
#             }
#             if self.prompt_flag:
#                 one_sample['prompt_flag'] = ann['prompt_flag']
#             return one_sample 
#         else:
#             one_sample = {
#                 "UserID": ann['UserID'],
#                 # "InteractedItemIDs_pad": None,
#                 # # "InteractedItemIDs": ann['InteractedItemIDs'],
#                 # "InteractedItemTitles": None,
#                 "TargetItemID": ann["TargetItemID"],
#                 "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
#                 # "InteractedNum": None,
#                 "label": ann['label']
#             }
#             if self.prompt_flag:
#                 one_sample['prompt_flag'] = ann['prompt_flag']
#             return one_sample 


class AmazonOOData(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None, seq_len=None, user2group='', use_ids=False, use_desc=True, sas_seq_len=None):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        ann_paths = ann_paths[0].split('=') 
        self.annotation = pd.read_pickle(ann_paths[0]+"_ood2.pkl").reset_index(drop=True)
        self.id2title = self.filter_dict(json.load(open('/'.join(ann_paths[0].split('/')[:-1])+'/id2title.json', 'r')))
        self.id2title[0] = {'title':'','keywords':''}
        # self.title2id = self.filter_dict(json.load(open('/'.join(ann_paths[0].split('/')[:-1])+'/title2id.json', 'r')))
        self.use_his = False
        self.use_label = False
        self.use_reason = False
        self.use_ids = use_ids
        self.use_desc = use_desc
        self.prompt_flag = False
        self.user2group = None
        self.sas_seq_len = sas_seq_len
        # user2group = f"{'/'.join(ann_paths[0].split('/')[:-1])}/user_group.csv"
        # print(user2group,os.path.exists(user2group))
        # if os.path.exists(user2group) and ('valid' in ann_paths[0] or 'test' in ann_paths[0]):
        if os.path.exists(user2group) and 'test' in ann_paths[0]:
        # if os.path.exists(user2group):
            self.user2group = pd.read_csv(user2group)[['user_id','cluster']]
            self.user2group.columns = ['user_id','group_id']
            self.user2group['user_id'] = self.user2group['user_id'].astype(int)
            self.user2group['group_id'] = self.user2group['group_id'].astype(int)
            print('load splits by group')
        else:
            print('no need to split by group')

        # ## warm test:
        
        if 'not_cold' in self.annotation.columns and "warm" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([1])].copy()
        if 'not_cold' in self.annotation.columns and "cold" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([0])].copy()

        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid','iid','his','label']
            renamed_columns = ['UserID','TargetItemID', 'InteractedItemIDs', 'label']
            # renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles', 'label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            # label of his
            if 'his_label' in self.annotation.columns:
                used_columns.append("his_label")
                renamed_columns.append("InteractedItemLabels")
                self.use_label = True
            if 'reason' in self.annotation.columns:
                used_columns.append("reason")
                renamed_columns.append("reason")
                self.use_reason = True
            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            if self.use_label:
                self.annotation["InteractedItemLabels"] = self.annotation["InteractedItemLabels"].map(list)
            
            # self.annotation["InteractedItemTitles"], self.annotation["InteractedItemDescs"] = self.turnid2txt(self.annotation["InteractedItemIDs"])
            # self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            # self.annotation["InteractedItemDescs"] = self.annotation["InteractedItemDescs"].map(list)
        else:
            used_columns = ['uid','iid','title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.items = list(set(self.annotation['TargetItemID']))
        self.text_processor = text_processor
        if self.user2group is not None:
            # print(type(self.annotation['UserID'][0]),type(self.user2group['user_id'][0]))
            # merged_df = pd.merge(self.annotation, self.user2group, left_on='UserID', right_on='user_id', how='left')
            # sorted_df = merged_df[merged_df['group_id']!=2]
            # print(sorted_df['group_id'].unique())
            # self.annotation = sorted_df.drop(columns=['user_id', 'group_id']).reset_index(drop=True)
            self.annotation = self.annotation.sort_values(by=['UserID','TargetItemID']).reset_index(drop=True)
            print('dataset reordered by UserID')

        if self.use_his:
            # 从self.annotation["InteractedItemIDs"]统计id热度
            # id_counter = Counter()
            # id_counter.update(self.annotation['UserID'])
            # self.id2hot = dict(id_counter)
            # self.id2hot_smoothed = defaultdict(int, {i: 0 for i in range(self.user_num)})
            # for user_id, count in self.id2hot.items():
            #     self.id2hot_smoothed[user_id] = np.log1p(count)#平滑热度
            # total_count = sum(self.id2hot_smoothed.values())
            # self.id2hot_smoothed = np.array(list(self.id2hot_smoothed.values()))
            # if total_count > 0:
            #     self.id2hot_smoothed /= total_count
            # self.annotation = self.annotation[self.annotation['InteractedItemIDs'].map(lambda x: len(x)) > 2]
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, seq_len) # average: only 50; 0915: 15 
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            # InteractedNum = len(a)
            # if a[0] == 0:
            #     InteractedNum -= 1
            # 找到第一个非零元素的索引
            first_non_zero_index = next((i for i, x in enumerate(a) if x != 0), len(a))
            # 更新 InteractedNum 为非零元素的数量
            InteractedNum = len(a) - first_non_zero_index
            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            if self.sas_seq_len is not None:
                if len(a) < self.sas_seq_len: # used for sasrec
                    c = [0]*(self.sas_seq_len - len(a))
                    c.extend(a)
                elif len(a) >= self.sas_seq_len:
                    c = a[-self.sas_seq_len:]

            if self.use_label:
                InteractedItemLabels = ann['InteractedItemLabels']
                if len(InteractedItemLabels)<self.max_lenght:
                    InteractedItemLabels_pad = [0]* (self.max_lenght-len(InteractedItemLabels))
                    InteractedItemLabels_pad.extend(InteractedItemLabels)
                elif len(InteractedItemLabels)> self.max_lenght:
                    InteractedItemLabels_pad = InteractedItemLabels[-self.max_lenght:]
                else:
                    InteractedItemLabels_pad = InteractedItemLabels
                InteractedItemLabels = InteractedItemLabels[-InteractedNum:]
            else:
                InteractedItemLabels = None
                InteractedItemLabels_pad = None
            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v4(samples=self.turnid2txt(ann["InteractedItemIDs"][-InteractedNum:]), labels=InteractedItemLabels, withid=self.use_ids),
                # "InteractedItemLabels_pad": np.array(InteractedItemLabels_pad),
                # "InteractedItemTitles": convert_title_list_v3(ann['InteractedItemTitles'][-InteractedNum:],ann['InteractedItemDescs'][-InteractedNum:],self.use_ids),
                # "CandidateItemIDs": np.array(candidates['CandidateItemIDs']),
                # "CandidateItemTitles": convert_title_list_v3(candidates['CandidateItemTitles'],candidates['CandidateItemDescs'],self.use_ids),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": convert_title_list_v4(samples=self.turnid2txt(ann["TargetItemID"]),withid=False),
                # "TargetItemTitle": f'"{ann["TargetItemTitle"]}"',
                # "TargetItemTitle": convert_title_list_v3([ann["TargetItemTitle"]],[ann["TargetItemDesc"]]),
                # "TargetItemTitle": "\""+ann["TargetItemTitle"].strip(' ')+"\"",
                # "TargetItemDesc": "\""+ann["TargetItemDesc"].strip(' ')+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label'],
                # "hot": 1/(1+self.id2hot_smoothed[ann['UserID']])
            }
            if InteractedItemLabels_pad:
                one_sample['InteractedItemLabels_pad'] = np.array(InteractedItemLabels_pad)
            if self.use_reason:
                one_sample['reason'] = ann['reason']
            else:
                one_sample['reason'] = 'Yes' if ann['label'] else 'No'
            if self.sas_seq_len is not None:
                one_sample['sas_seq'] = np.array(c)
        else:
            one_sample = {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                # "InteractedNum": None,
                "label": ann['label']
            }
        if self.prompt_flag:
            one_sample['prompt_flag'] = ann['prompt_flag']
        return one_sample 
    
    def turnid2txt(self,ids):
        if isinstance(ids,(int,np.integer)):
            ids = [ids]
        elif isinstance(ids,list):
            pass
        else:
            try :
                ids = list(ids)
            except:
                raise NotImplementedError(f"Unsupported type of ids {type(ids)}")
        titles = [self.id2title[idx]['title'] for idx in ids]
        descriptions = None
        if self.use_desc:
            descriptions = [self.id2title[idx]['keywords'] for idx in ids]
        return titles, descriptions

    def get_id2title(self):
        return self.id2title

    # def get_title2id(self):
    #     return self.title2id

    def filter_dict(self,raw_dict):
        filtered_dict = {}
        for item_id, details in raw_dict.items():
            if 'title' in details and 'keywords' in details:
                filtered_dict[int(item_id)] = {
                    'title': details['title'],
                    'keywords': details['keywords']
                }
            if 'id' in details and 'keywords' in details:
                filtered_dict[item_id] = {
                    'id': details['id'],
                    'keywords': details['keywords']
                }
        return filtered_dict

# class AmazonOOData_sasrec(RecBaseDataset):
#     def __init__(self, text_processor=None, ann_paths=None, seq_len=None, sas_seq_len=20):
#         super().__init__()
#         # self.vis_root = vis_root
#         # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
#         self.annotation = pd.read_pickle(ann_paths[0]+"_ood2.pkl").reset_index(drop=True)
        
#         self.use_his = False
#         self.prompt_flag = False
#         self.sas_seq_len = sas_seq_len

#         if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
#             used_columns = ['uid','iid','title','his', 'his_title','label']
#             renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
#             if 'not_cold' in self.annotation.columns:
#                 used_columns.append("not_cold")
#                 renamed_columns.append("prompt_flag")
#                 self.prompt_flag = True

#             self.use_his = True
#             self.annotation = self.annotation[used_columns]
#             self.annotation.columns = renamed_columns
        
#             self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
#             self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
#             self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"] #.map(convert_title_list)
#         else:
#             used_columns = ['uid','iid','title','label']
#             renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']
#             if 'not_cold' in self.annotation.columns:
#                 used_columns.append('not_cold')
#                 renamed_columns.append("prompt_flag")
#                 self.prompt_flag = True
#             self.annotation = self.annotation[used_columns]
#             self.annotation.columns = renamed_columns
        
#         print("data path:", ann_paths[0], "data size:", self.annotation.shape)
#         self.user_num = self.annotation['UserID'].max()+1
#         self.item_num = self.annotation['TargetItemID'].max()+1
#         self.text_processor = text_processor
        
        
#         if self.use_his:
#             max_length_ = 0
#             for x in self.annotation['InteractedItemIDs'].values:
#                 max_length_ = max(max_length_, len(x))
#             self.max_lenght = min(max_length_, seq_len) # average: only 50; 0915: 15 
#             print("Movie OOD datasets, max history length:", self.max_lenght)
#             logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))
            
#     def __getitem__(self, index):

#         # TODO this assumes image input, not general enough
#         ann = self.annotation.iloc[index]
#         if self.use_his:
#             a = ann['InteractedItemIDs']
#             InteractedNum = len(a)
#             if a[0] == 0:
#                 InteractedNum -= 1

#             if len(a) < self.max_lenght:
#                 b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
#                 b.extend(a)
#             elif len(a)> self.max_lenght:
#                 b = a[-self.max_lenght:]
#                 InteractedNum = self.max_lenght
#             else:
#                 b = a
            
#             if len(a) < self.sas_seq_len: # used for sasrec
#                 c = [0]*(self.sas_seq_len - len(a))
#                 c.extend(a)
#             elif len(a) >= self.sas_seq_len:
#                 c = a[-self.sas_seq_len:]

#             one_sample = {
#                 "UserID": ann['UserID'],
#                 "InteractedItemIDs_pad": np.array(b),
#                 "InteractedItemIDs": ann['InteractedItemIDs'],
#                 "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
#                 "TargetItemID": ann["TargetItemID"],
#                 "TargetItemTitle": "\""+ann["TargetItemTitle"].strip(' ')+"\"",
#                 "InteractedNum": InteractedNum,
#                 "label": ann['label'],
#                 "sas_seq": np.array(c)
#             }
#             if self.prompt_flag:
#                 one_sample['prompt_flag'] = ann['prompt_flag']
#             return one_sample 
#         else:
#             one_sample = {
#                 "UserID": ann['UserID'],
#                 # "InteractedItemIDs_pad": None,
#                 # # "InteractedItemIDs": ann['InteractedItemIDs'],
#                 # "InteractedItemTitles": None,
#                 "TargetItemID": ann["TargetItemID"],
#                 "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
#                 # "InteractedNum": None,
#                 "label": ann['label']
#             }
#             if self.prompt_flag:
#                 one_sample['prompt_flag'] = ann['prompt_flag']
#             return one_sample 









class MovielensDataset(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__(text_processor, ann_paths)
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0]+".pkl").reset_index(drop=True)
        self.use_his = False
        if 'sessionItems' in self.annotation.columns:
            self.use_his = True
            self.annotation = self.annotation[['uid','iid','title','sessionItems', 'sessionItemTitles','label']]
            self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(convert_title_list)
        else:
            self.annotation = self.annotation[['uid','iid','title','label']]
            self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle','label']
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
        if self.use_his:
            max_length = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length = max(max_length,len(x))
            self.max_lenght = max_length
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            else:
                b = a
            return {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": ann['InteractedItemTitles'],
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"]+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
        else:
            return {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"]+"\"",
                # "InteractedNum": None,
                "label": ann['label']
            }


class MovielensDataset_stage1(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__(text_processor, ann_paths)
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0]+".pkl").reset_index(drop=True)[['uid','iid','title','sessionItems', 'sessionItemTitles','label', 'pairItems', 'pairItemTitles']]
        self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label','PairItemIDs','PairItemTitles']
        self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
        self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
        self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(convert_title_list)
        self.annotation["PairItemTitles"] = self.annotation["PairItemTitles"].map(convert_title_list)
        
        
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        return {
            "UserID": ann['UserID'],
            "PairItemIDs": np.array(ann['PairItemIDs']),
            "PairItemTitles": ann["PairItemTitles"],
            "label": ann['label']
        }


class AmazonDataset(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0]+"_seqs.pkl").reset_index(drop=True)
        self.use_his = False
        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            self.use_his = True
            self.annotation = self.annotation[['uid','iid','title','his', 'his_title','label']]
            self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"] #.map(convert_title_list)
        else:
            self.annotation = self.annotation[['uid','iid','title','label']]
            self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle','label']
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_,len(x))
            self.max_lenght = min(max_length_, 15) # average: only 5 
            print("amazon datasets, max history length:", self.max_lenght)
            logging.info("amazon datasets, max history length:" + str(self.max_lenght))
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            return {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"]+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
        else:
            return {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"],
                # "InteractedNum": None,
                "label": ann['label']
            }
