from calendar import c
import logging
import random
import time

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F
import torch.nn as nn
import os

from minigpt4.common.registry import registry
from minigpt4.models.rec_model import Rec2Base, disabled_train
from transformers import LlamaTokenizer, GenerationConfig, AutoTokenizer, AutoConfig
import re
import numpy as np
import pandas as pd
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, set_peft_model_state_dict

def get_ids_order(prompt):
    id_flags = ["<UserID>", "<HisItemList>", "<TargetItemID>", "<TargetItemList>"]
    id_order_ = []
    for flag_ in id_flags:
        pos_ = prompt.find(flag_)
        if pos_>=0:
            id_order_.append(pos_)
    id_order_ = np.argsort(np.array(id_order_))
    # print('prompt',prompt,'\nid_order_',id_order_)
    return id_order_

def consitence_loss(ori_embs, proj_embs):
    ori_embs = ori_embs.squeeze()
    proj_embs = proj_embs.squeeze()
    ori_similarities = torch.matmul(ori_embs, ori_embs.T)
    # ori_diag = torch.diag(ori_similarities)+1e9
    proj_similarities = torch.matmul(proj_embs, proj_embs.T)
    # proj_diag = torch.diag(proj_similarities)+1e9
    N_ = ori_similarities.shape[0]
    ori_similarities[range(N_), range(N_)] -= 1e9
    proj_similarities[range(N_), range(N_)] -= 1e9
    ori_similarities = torch.softmax(ori_similarities,dim=-1) 
    proj_similarities = torch.softmax(proj_similarities,dim=-1)
    loss = nn.functional.mse_loss(ori_similarities, proj_similarities)
    # loss = -torch.log(proj_similarities+1e-6).mul(ori_similarities).sum(dim=-1).mean() #+ nn.functional.cross_entropy(,)
    # loss = nn.functional.kl_div(proj_similarities, ori_similarities, reduction="batchmean")
    return loss 

class identical_map(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self,x):
        return x*1.0


@registry.register_model("mini_gpt4rec_v3")
class MiniGPT4Rec_v3(Rec2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4rec.yaml",
    }

    def __init__(
        self,
        rec_model="MF",
        rec_config=None,
        pretrained_rec=None,
        freeze_rec=True,
        rec_precision='fp16',
        infer_type='native',
        quest_config=None,
        cake_config=None,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        proj_token_num=1, # the number of tokens that the user/item embedding projected to
        proj_drop=0,
        lora_config=None,
        loss_config=None,
        proj_mid=5,
        freeze_lora=False,
        freeze_proj=False
    ):
        super().__init__()

        # self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.proj_token_num = proj_token_num
        self.use_quest = False
        self.use_cake = False

        print("runing MiniGPT4Rec_v3 ...... ")

        print('Loading Rec_model')
        self.rec_model_type = rec_model
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config, rec_precision)
        # try:
        if self.rec_encoder is not None and pretrained_rec != "not_have":
            self.rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location="cpu"))
            print("successfully load the pretrained model......")
        # except:
        #     # print(pretrained_rec)
        #     # self.rec_encoder.config
        #     raise RuntimeError("Please provide your pretained rec model path or check whether the pretrained model and the defined mode can match each other")
        if freeze_rec and self.rec_encoder is not None:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train
            logging.info("freeze rec encoder")
            print("freeze rec encoder")

        print('Loading Rec_model Done')

            

        print('Loading LLAMA')
        if "Qwen" in llama_model:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, trust_remote_code=True)
            self.llama_tokenizer.unk_token = "<|object_ref_end|>"
            self.llama_tokenizer.unk_token_id = 151647
        else:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            if self.llama_tokenizer.unk_token is None:
                self.llama_tokenizer.add_special_tokens({"unk_token":"<unk>"})

        
        if infer_type == 'quest' and quest_config is not None:
            from quest import LlamaForCausalLM
            self.use_quest = True
            attn_implementation=None

        elif infer_type == 'cake' and cake_config is not None:
            self.use_cake = True
            from cake.model.modeling_llama import LlamaForCausalLM
            from cake.cake_cache import CakeprefillKVCache
            from cake.utils import CompressConfig
            from cake.monkeypatch import replace_flashllama_attn_with_cakeattn

            compress = cake_config.compress
            cascading = cake_config.cascading
            compress_config = CompressConfig(compress, cascading)
            compress_config.cache_size = cake_config.cache_size
            compress_config.window_size = cake_config.window_size
            # hyper = [1.6, 0.4, 200.0]
            compress_config.hyper = [cake_config.tau1, cake_config.tau2, cake_config.gamma]

            replace_flashllama_attn_with_cakeattn()
            attn_implementation="flash_attention_2"
        else:
            from transformers import LlamaForCausalLM
            attn_implementation=None

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                attn_implementation=attn_implementation
            )

        if self.use_cake:
            model_config = AutoConfig.from_pretrained(llama_model)
            layers = model_config.num_hidden_layers
            for i in range(layers):
                self.llama_model.model.layers[i].self_attn.config.key_size = [compress_config.cache_size - compress_config.window_size]*layers
                self.llama_model.model.layers[i].self_attn.config.window_size = [compress_config.window_size]*layers
                self.llama_model.model.layers[i].self_attn.config.prefill = [True]*layers
                self.llama_model.model.layers[i].self_attn.config.decoding_evict = [None]*layers
                self.llama_model.model.layers[i].self_attn.config.tau1 = compress_config.hyper[0]
                self.llama_model.model.layers[i].self_attn.config.tau2 = compress_config.hyper[1] 
                self.llama_model.model.layers[i].self_attn.config.gamma = compress_config.hyper[2] 
                self.llama_model.model.layers[i].self_attn.config.prefill_cake_evict = [CakeprefillKVCache(
                    cache_size=compress_config.cache_size,
                    window_size=compress_config.window_size,
                    k_seq_dim=2,
                    v_seq_dim=2,
                    num_heads=self.llama_model.model.layers[i].self_attn.num_heads,
                    num_layers=layers,
                    use_cascading=compress_config.cascading
                )]*layers

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.use_lora = False
        if lora_config is not None and lora_config.use_lora:
            print("Setting Lora")
            self.use_lora = True
            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llama_model_lora = get_peft_model(self.llama_model, peft_config)
            print("Setting Lora Done")
        
        if freeze_lora:
            print("freeze lora...")
            try:
                flexible_layer_num = freeze_lora.layers
                flexible_layers = range(32-flexible_layer_num,32)
            except:
                flexible_layers = None
            for name, param in self.llama_model_lora.named_parameters():
                if flexible_layers and 'layers' in name and 'lora_' in name:
                    layer_id = int(re.search(r'\.layers\.([^.]+)', name).group(1))
                    if layer_id in flexible_layers:
                        param.requires_grad = True
                        continue
                param.requires_grad = False

        if self.use_quest:
            print("Setting Quest")
            self.llama_model.quest_init(page_size=quest_config.page_size, max_seq_len=quest_config.max_seq_len, token_budget=quest_config.token_budget)
            if self.use_lora:
                self.llama_model_lora.quest_init(page_size=quest_config.page_size, max_seq_len=quest_config.max_seq_len, token_budget=quest_config.token_budget)
            print("Setting Quest Done")
        
        if self.rec_encoder is not None and 'prompt' not in rec_model:
            print("proj_mid:", type(proj_mid), proj_mid)
            self.llama_proj = nn.Sequential(
                nn.Linear(self.rec_encoder.config.embedding_size, self.rec_encoder.config.embedding_size*int(proj_mid)),  # ml100=>5
                nn.ReLU(),
                # nn.Dropout(proj_drop),
                nn.Linear(self.rec_encoder.config.embedding_size*int(proj_mid), self.llama_model.config.hidden_size * self.proj_token_num),
            )
            # self.llama_proj = nn.Linear(self.rec_encoder.config.embedding_size, self.llama_model.config.hidden_size * self.proj_token_num)
        elif self.rec_encoder is not None and rec_model=="personlized_prompt": #'prompt' in rec_model:
            # identical mapping function, i.e., f(x)=x
            print("personalized prompt learning....")
            self.llama_proj = nn.Linear(rec_config.item_num+rec_config.user_num, self.llama_model.config.hidden_size * self.proj_token_num,bias=False) #identical_map()
        elif self.rec_encoder is not None and rec_model=="soft_prompt": #'prompt' in rec_model:
            # identical mapping function, i.e., f(x)=x
            print("soft prompt learning....")
            self.llama_proj = nn.Linear(2, self.llama_model.config.hidden_size * self.proj_token_num,bias=False) #identical_map()
        else:
            self.llama_proj = None
        
        if freeze_proj:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            self.llama_proj = self.llama_proj.eval()
            self.llama_proj.train = disabled_train
            logging.info("!!!! freeze llama_proj...")

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.has_print_prompt=False

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt List: \n{}'.format(self.prompt_list))
            self.has_pri_decode=False
            self.prompt_list_p = None
        else:
            self.prompt_list = []
            self.prompt_list_p = None
        self.reason =  False
        self.eval_only = False
        self.user2group = None
        self.wrong_data = 0
        self.wrong_dir = '/home/yuqihang/projects/CoLLM/collm-datasets/bookdu/reflection/wrong.txt'
        self.loss_alpha = 1.0
        self.loss_beta = 0
        self.loss_theta = 1.0
        self.loss_gamma = 0
        if loss_config is not None:
            try:
                self.loss_alpha = loss_config.alpha
                self.loss_beta = loss_config.beta
                self.loss_theta = loss_config.theta
                self.loss_gamma = loss_config.gamma
            except:
                print('using default loss_config')

    def set_generate_config(self,generate_config,ckpt_path):
        if generate_config is not None and generate_config.enable:
            self.eval_only = True
            self.generate_length = generate_config.max_len
            if not isinstance(ckpt_path,str):
                ckpt_path = ckpt_path[0]
            self.generete_file = ckpt_path.replace('.pth','')+'.txt'

    def set_user2group(self, user2group):
        print(user2group)
        if os.path.exists(user2group):
            self.user2group = pd.read_csv(user2group)
            self.user2group.columns = ['user_id','group_id']
            self.user2group['user_id'] = self.user2group['user_id'].astype(int)
            self.user2group['group_id'] = self.user2group['group_id'].astype(int)
            print('set user2group done!!!')

    def to_be_trained(self):
        if self.use_lora:
            return True
        # return True # have lora module, will be trained anyway
        id_terms = ["<UserID>", "<HisItemTitleList>", "<TargetItemTitleList>", "<HisItemList>", "TargetItemList"]
        # id_terms = ["<UserID>", "<ItemIDList>", "<TargetItemID>", "<DCNFeature>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    return True
        ### No ID is used, disable the projection layers
        # self.llama_proj = None
        # for name, param in self.llama_proj.named_parameters():
        #     param.requires_grad = False  
        return False
    
    def set_mode(self, mode):
        '''
        mode \in ['v1','v2','v3',None]
        '''
        self.run_mode_ = mode
    
    def rec_to_cpu(self):
        self.rec_encoder.to("cpu")
        self.rec_encoder.float()
    
    def set_answer_type(self,mode):
        if mode == 'v1':
        # pos_ans = ["The former item.", "The first item.", "The former.", "The first.", "The former one.", "The first one."]
        # neg_ans = ["The latter item.", "The second item.", "The latter.", "The second.", "The latter one.", "The second one."]
            self.pos_ans = ["former"]
            self.neg_ans = ["latter"]
        elif mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            # self.pos_ans = ['enjoy']
            # self.neg_ans = ['dislike']
            pos_ans_id = self.llama_tokenizer(self.pos_ans[0],add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(self.neg_ans[0],add_special_tokens=False).input_ids[0]
            print("answer token ids: pos:",pos_ans_id, "neg ids:", neg_ans_id)
        elif mode == 'v3':
            print("answer is concrect item title.")
        else:
            raise NotImplementedError("not implement this types of answers")
    def print_prompt(self):
        pass
        # print('Prompt Pos Example \n{} {} or {}'.format(random.choice(self.prompt_list),self.pos_ans[0],self.neg_ans[0]))

    def forward_v3(self, samples):
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_with_p([5,5,5,1])) #[1,5,3,1]  #[2,5,3,1]
            sample_embeds, sample_atts = self.prompt_based_encode_v3(prompt,samples)
        self.llama_tokenizer.padding_side = "right"
        device = samples['UserID'].device #samples_encode['User_emb'].device
        if "reason" in samples :
            self.reason = True

        ans_ = {1:self.pos_ans[0], 0:self.neg_ans[0]}
        pos_ans_id = self.llama_tokenizer(ans_[int(1)],add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(ans_[int(0)],add_special_tokens=False).input_ids[0]

        if self.reason:
            text = samples["reason"]
            # print("reason:",text)
        else:
            text = [ans_[int(t)] for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([sample_atts.shape[0],sample_atts.shape[1]],dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([sample_atts, to_regress_tokens.attention_mask], dim=1)

        if self.user2group is not None:
            current_adapter = self.llama_model_lora.active_adapter #group0,group1
            group_id = f"group{self.user2group[self.user2group['user_id']==samples['UserID'].cpu().tolist()[0]]['group_id'].item()}"
            # print(current_adapter,group_id)
            if group_id != current_adapter:
                # print(f'switch to {group_id}')
                self.llama_model_lora.set_adapter(group_id)
            
        print(f'{inputs_embeds.shape[1]}',file=open('ftoken_lens.txt','a'))
        with self.maybe_autocast():
            outputs = self.llama_model_lora(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        # new loss, just focus on the target pos and neg tokens 

        logits = outputs.logits[:,-t_posi,:][:,pos_ans_id]
        loss_pred = nn.functional.binary_cross_entropy_with_logits(logits, samples['label'].float()) #+ 1e-7 * samples_encode['loss_c']

        if t_posi>=3:# reason 2 4500
            # print(f'shapes:{loss_pred.shape} {outputs.loss.shape}\nloss_pred{loss_pred},loss{outputs.loss}')
            loss = self.loss_alpha * loss_pred + self.loss_beta * outputs.loss
        else:
            loss = self.loss_theta * loss_pred + self.loss_gamma * outputs.loss
        return {"loss": loss}

    def generate_for_samples_v3(self, samples, return_all=False):
        ret = {}
        # sample = samples["image"]
        user_selective_prompts = False
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
            # vqa_prompt = '###Human: <Img><ImageHere></Img> '
            # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = self.prompt_list[0]
            # sample_embeds, sample_atts, reflect_embeds, reflect_atts = self.prompt_based_encode_v3(prompt,samples,True)
            sample_embeds, sample_atts = self.prompt_based_encode_v3(prompt,samples,False)

        self.llama_tokenizer.padding_side = "right"

        device = samples['UserID'].device #samples_encode['User_emb'].device

        ans_ = {1:self.pos_ans[0], 0:self.neg_ans[0]}
        start_time = time.time()
        if not self.eval_only:
            text = [ans_[int(t)]  for t in samples["label"]]
            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(device)
            t_posi = to_regress_tokens.input_ids.shape[-1] + 1
            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )
            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            empty_targets = torch.ones([sample_atts.shape[0],sample_atts.shape[1]],dtype=torch.long).to(device).fill_(-100)
            inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
            inputs_atts = torch.cat([sample_atts, to_regress_tokens.attention_mask], dim=1)
            QAtargets = torch.cat([empty_targets, targets], dim=1)
            print(f'{inputs_embeds.shape[1]}',file=open('gtoken_lens.txt','a'))

        if self.user2group is not None:
            current_adapter = self.llama_model_lora.active_adapter #group0,group1
            group_id = f"group{self.user2group[self.user2group['user_id']==samples['UserID'].cpu().tolist()[0]]['group_id'].item()}"
            # print(current_adapter,group_id)
            if group_id != current_adapter:
                print(f'switch to {group_id}')
                self.llama_model_lora.set_adapter(group_id)

        with self.maybe_autocast():
            if self.eval_only:
                outputs = self.llama_model_lora.generate(
                    inputs_embeds=sample_embeds,
                    attention_mask=sample_atts,
                    temperature=0.4,
                    no_repeat_ngram_size = 2,
                    repetition_penalty=1.2,
                    # do_sample=False,
                    max_length=self.generate_length + sample_embeds.shape[-2],
                    early_stopping=False, # 禁用提前停止
                    pad_token_id=self.llama_tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    use_cache=True,
                    output_logits=True,
                )
                if self.use_quest:
                    self.llama_model_lora.quest_clear()
                if self.use_cake:
                    layers = len(self.llama_model_lora.base_model.model.model.layers)
                    for i in range(layers):
                        self.llama_model_lora.base_model.model.model.layers[i].self_attn.config.prefill = [True]*layers
                        self.llama_model_lora.base_model.model.model.layers[i].self_attn.config.decoding_evict = [None]*layers

            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs_atts,
                    return_dict=True,
                    use_cache=False,
                    labels=QAtargets,
                )
        end_time = time.time()
        ret['cost'] = end_time - start_time
        pos_ans_id = self.llama_tokenizer(ans_[int(1)], add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(ans_[int(0)], add_special_tokens=False).input_ids[0]
        if self.eval_only:
            # ret['outputs'] = outputs
            reason_text = self.llama_tokenizer.batch_decode(**outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ret['reason_text'] = reason_text
            #存储生成的文本
            print(samples['label'].cpu().tolist(),file=open(self.generete_file,'a'))
            print('\n'.join(['. '.join(ana.replace('\n',' ').split('. ')[:-1]) for ana in reason_text]),file=open(self.generete_file,'a'))
            # print('\n'.join([ana.replace('\n',' ') for ana in self.llama_tokenizer.batch_decode(**outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)]),file=open(self.generete_file,'a'))
            logits_ = torch.stack(outputs.logits, dim=1)[:,0,:][:,pos_ans_id]
        else:
            logits_ = outputs.logits[:,-t_posi,:][:,pos_ans_id]
        # logits_ = logits[:,0,:][:,pos_ans_id]-logits[:,0,:][:,neg_ans_id]
        loss = nn.functional.binary_cross_entropy_with_logits(logits_, samples['label'].float())
        # self.save_wrong_data(samples, logits_)
        ret["loss"] = loss
        ret["logits"] = logits_
        return ret

    def save_wrong_data(self, samples, logits ):
        logits[logits>0.5] = 1
        logits[logits<=0.5] = 0
        logits = torch.tensor(logits,dtype=torch.int8).cpu().numpy()
        labels = torch.tensor(samples["label"],dtype=torch.int8).cpu().numpy()

        UserID = samples['UserID'].cpu().numpy()
        TargetItemID = samples['TargetItemID'].cpu().numpy()
        InteractedItemIDs_pad = samples['InteractedItemIDs_pad'].cpu().numpy()
        InteractedNum = samples['InteractedNum'].cpu().numpy()
        InteractedItemLabels = samples['InteractedItemLabels_pad'].cpu().numpy()
        InteractedItemTitles = samples['InteractedItemTitles']
        TargetItemTitle = samples['TargetItemTitle']
        print(self.wrong_data,logits,labels)
        for idx,logit in enumerate(logits):
            if logit != labels[idx]:
                self.wrong_data += 1
                print(f'{UserID[idx]}\sep{TargetItemID[idx]}\sep{list(InteractedItemIDs_pad[idx][-InteractedNum[idx]:])}\sep{labels[idx]}\sep{list(InteractedItemLabels[idx][-InteractedNum[idx]:])}\sep{InteractedItemTitles[idx]}\sep{TargetItemTitle[idx]}',file=open(self.wrong_dir,'a'))


    def decode_logits_to_text(self, logits, t_posi=None):
        # 对最后一个维度进行softmax操作
        probs = F.softmax(logits, dim=-1)  # shape: [batch_size, sequence_length, vocab_size]
        
        # 获取每个位置上概率最大的token ID
        predicted_token_ids = torch.argmax(probs, dim=-1)  # shape: [batch_size, sequence_length]
        
        # Step 2: 使用tokenizer将token ID转换为文本
        decoded_texts = []
        for batch_idx in range(predicted_token_ids.shape[0]):
            # 获取当前样本的 token IDs
            token_ids = predicted_token_ids[batch_idx].tolist()
            
            # 如果只需要解码特定位置的token，可以只取该位置的token ID
            if t_posi is not None:
                token_ids = [token_ids[-t_posi]]
            
            # 使用 tokenizer 解码 token IDs
            decoded_text = self.llama_tokenizer.decode(token_ids, skip_special_tokens=True)
            
            # 添加到结果列表中
            decoded_texts.append(decoded_text)
        
        return decoded_texts

    def prompt_based_encode_v3(self, prompt, samples, reflect=False):
        id_orders = get_ids_order(prompt)
        samples_encode, sample_atts = self.encode_recdata_v3(samples,ids_order=id_orders)
        sample_embeds, samples_atts = self.recprompt_wrap_v3(samples_encode, samples, sample_atts, prompt)
        if reflect:
            reflect_embeds, reflect_atts = self.reflect_prompt_wrap(samples_encode, samples, None)
            return sample_embeds, samples_atts, reflect_embeds, reflect_atts
        return sample_embeds, samples_atts


    def encode_recdata_v3(self, sample, ids_order=None):  # used for stage2
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        
        with self.maybe_autocast():
            batch_size = sample['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            if self.rec_model_type == "sasrec":  # for sasrec, there is no user encoder but just seqs encoder, we take it to get user representation
                user_embeds = self.rec_encoder.seq_encoder(sample['sas_seq']).unsqueeze(-2)
            elif self.rec_model_type == "DCN" or self.rec_model_type == "DIN":
                """
                not really user embeding, but the embedding merged for one sample point
                """
                user_embeds = self.rec_encoder.all_encode(sample['UserID'],sample['TargetItemID'],sample['sas_seq'][:,-10:]).unsqueeze(-2)
            else:
                user_embeds = self.rec_encoder.user_encoder(sample['UserID'], all_users=all_user_embeds).unsqueeze(-2)
            # ***Note: here, for sasrec, item embedding comes form the last layer 
            targetItem_embed = self.rec_encoder.item_encoder(sample['TargetItemID'], all_items=all_item_embeds).unsqueeze(-2)

            user_embeds_llama = self.llama_proj(user_embeds).reshape(batch_size,-1, self.proj_token_num, hidden_size)
            targetItem_embeds_llama = self.llama_proj(targetItem_embed).reshape(batch_size,-1, self.proj_token_num, hidden_size)
            
            # loss_c = consitence_loss(user_embeds, user_embeds_llama) + consitence_loss(targetItem_embed, targetItem_embeds_llama)
            # print('which',len(ids_order),sample.keys())
            if 'InteractedItemIDs_pad' in sample.keys() and len(ids_order)==3:
                interactedItem_embeds = self.rec_encoder.item_encoder(sample['InteractedItemIDs_pad'], all_items=all_item_embeds)
                interactedItem_embeds_llama = self.llama_proj(interactedItem_embeds).reshape(batch_size,-1, self.proj_token_num, hidden_size)
                # candidateItem_embeds = self.rec_encoder.item_encoder(sample['CandidateItemIDs'], all_items=all_item_embeds)
                # candidateItem_embeds_llama = self.llama_proj(candidateItem_embeds).reshape(batch_size,-1, self.proj_token_num, hidden_size)

                # merged_embeds = [user_embeds_llama, interactedItem_embeds_llama, candidateItem_embeds_llama]
                merged_embeds = [user_embeds_llama, interactedItem_embeds_llama, targetItem_embeds_llama]
                merged_embeds = [merged_embeds[k] for k in ids_order]
                merged_embeds = torch.cat(merged_embeds,dim=1)
                idx_flag = torch.ones_like(sample['InteractedItemIDs_pad'])
                idx_flag = torch.where(sample['InteractedItemIDs_pad']==self.rec_encoder.padding_index, 0, idx_flag) # indx_of_paddded historical items
                # to indicate user_id, his_items_id, target_item_id
                # idx_flag = [torch.ones([idx_flag.shape[0],1]).to(idx_flag.device),idx_flag,torch.ones_like(sample['CandidateItemIDs']).to(idx_flag.device)]
                idx_flag = [torch.ones([idx_flag.shape[0],1]).to(idx_flag.device),idx_flag,torch.ones([idx_flag.shape[0],1]).to(idx_flag.device)]
                idx_flag = [idx_flag[k] for k in ids_order]
                idx_flag = torch.cat(idx_flag,dim=1).to(device)
                idx_nopad = torch.nonzero(idx_flag)
                
                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'InteractedItems_embs': interactedItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    # 'CandidateItems_embs': candidateItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'merged_embs': merged_embeds[idx_nopad[:,0],idx_nopad[:,1]].reshape(-1, hidden_size),
                    # 'loss_c': loss_c
                }
            else:
                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'InteractedItems_embs': None,
                    # 'CandidateItems_embs': None,
                    'merged_embs': None,
                    # 'loss_c': loss_c
                }
        sample_atts_llama = None
        # {
        #     'user': atts_user,
        #     'TargetItem': atts_targetItem,
        #     'InteractedItems': atts_interactedItem
        # }
        return sample_embeds_llama, sample_atts_llama


    def recprompt_wrap_v3(self, samples, ori_samples, atts_sample, prompt): # used for stage 2
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<UserID>", "<HisItemTitleList>", "<TargetItemTitleList>", "<HisItemList>", "<TargetItemList>", "<ItemID>"]
            # split_symbol = ["<UserID>", "<ItemIDList>", "<ItemTitleList>", "<TargetItemID>", "<TargetItemTitle>"]
            batch_size = ori_samples['UserID'].shape[0]
            bos = "<s>"
            if self.llama_tokenizer.unk_token:
                unk_token_id = self.llama_tokenizer.unk_token_id
                unk_ = self.llama_tokenizer.unk_token #"<unk>"
                # print('unk_token_id,unk_',unk_token_id,unk_)
            unk_ = ".".join([unk_]*self.proj_token_num)
            prompt = bos + prompt # add the bos
            prompt = prompt.replace("<UserID>", unk_)
            # prompt = prompt.replace("<TargetItemID>", unk_)

            # prompt = prompt.replace("<DCNFeature>", unk_)

            # interactedItems = samples['InteractedItemTitles']
            label2answer = {0:'No',1:'Yes'}
            prompt_list = []
            # answer_list = []
            
            for k in range(batch_size):
                prompt_ = prompt+""
                # prompt_ = prompt.replace('UserID',unk_)
                # item_num = samples['interacted']
                if 'InteractedNum' in ori_samples.keys():
                    prompt_ = prompt_.replace('<HisItemTitleList>', ori_samples['InteractedItemTitles'][k])
                    prompt_ = prompt_.replace('<HisItemList>', ori_samples['InteractedItemTitles'][k])
                    prompt_ = prompt_.replace('<ItemTitleList>', ori_samples['InteractedItemTitles'][k])
                # prompt_ = prompt_.replace("<TargetItemTitleList>", ori_samples['CandidateItemTitles'][k])
                # prompt_ = prompt_.replace("<TargetItemList>", ori_samples['CandidateItemTitles'][k])

                prompt_ = prompt_.replace("<TargetItemTitle>", ori_samples['TargetItemTitle'][k])
                prompt_ = prompt_.replace("<ItemID>", unk_)
                prompt_ = prompt_.replace("<TargetItemID>", unk_)
                # prompt_ += samples['Response'][k]
                prompt_list.append(prompt_)
                # answer_list.append(label2answer[ori_samples['label'][k].item()])
            
            if not self.has_print_prompt:
                print("prompt example:", prompt_list[0])
                # print("prompt example:", prompt_list[0], answer_list[0])
                # print("prompt example:", random.choice(prompt_list))
                self.has_print_prompt = True
            
            self.llama_tokenizer.padding_side = "left"
            prompts_tokens = self.llama_tokenizer(
                prompt_list,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(ori_samples['UserID'].device)

            # answer_tokens = self.llama_tokenizer(
            #     answer_list,
            #     return_tensors="pt",
            #     padding="longest",
            #     truncation=True,
            #     max_length=self.max_txt_len,
            #     add_special_tokens=False
            # ).to(ori_samples['UserID'].device)

            if not self.has_pri_decode:
                print("#######prmpt decoded example: ",' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))
                self.has_pri_decode = True
                
            # print(f"{max([len(prompt_) for prompt_ in prompt_list])}",file=open('text_lens.txt','a'))
            replaced_idx = torch.nonzero(prompts_tokens.input_ids==unk_token_id)

            prompt_embeds = self.llama_model_lora.base_model.model.model.embed_tokens(prompts_tokens.input_ids)
            # answer_embeds = self.llama_model_lora.base_model.model.model.embed_tokens(answer_tokens.input_ids)
            
            # prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = samples['merged_embs']
            # if "<UserID>" in prompt_ori  and "<HisItemList>" in prompt_ori and  "<TargetItemList>" in prompt_ori:
            try:
                if "<UserID>" in prompt_ori  and "<HisItemList>" in prompt_ori and  "<TargetItemID>" in prompt_ori:
                    if replaced_idx.shape[0]==2*samples['User_emb'].shape[0]:
                        prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = torch.cat([samples['User_emb'], samples['TargetItem_emb']],dim=-2).reshape(-1,samples['User_emb'].shape[-1])
                    else:
                        prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = samples['merged_embs']
                elif "<UserID>" in prompt_ori and "<TargetItemID>" in prompt_ori and "<HisItemList>" not in prompt_ori:
                    prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = torch.cat([samples['User_emb'], samples['TargetItem_emb']],dim=-2).reshape(-1,samples['User_emb'].shape[-1])
                else:
                    pass
            except:
                print(replaced_idx.shape[0],'sample',samples['User_emb'].shape[0],samples['TargetItem_emb'].shape[0])
                if samples['merged_embs'] is not None:
                    print(samples['merged_embs'].shape)
                print("#######prmpt decoded example: ",' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))
            # input_embeds = torch.cat([prompt_embeds, answer_embeds], dim=-2)
            # attention_mask = torch.cat([prompts_tokens.attention_mask, answer_tokens.attention_mask], dim=-1)
            # labels = torch.cat([torch.full_like(prompts_tokens.input_ids, -100), answer_tokens.input_ids.masked_fill(answer_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100)], dim=-1)
            return prompt_embeds, prompts_tokens.attention_mask


    def reflect_prompt_wrap(self, samples, ori_samples, prompt):
        prompt_ori = "Based on the descriptions and the user's enjoyment of each book in the historical sequence, construct a persona of the user's preferences and reevaluate whether the user would enjoy the book titled <TargetItemTitle>. Please begin your analysis with \"Yes\" or \"No\". \n#Answer:"
        batch_size = ori_samples['UserID'].shape[0]
        bos = "<s>"
        unk_ = self.llama_tokenizer.unk_token #"<unk>"
        unk_ = ".".join([unk_]*self.proj_token_num)
        prompt = bos + prompt_ori # add the bos

        prompt_list = []
        
        for k in range(batch_size):
            prompt_ = prompt+""
            prompt_ = prompt_.replace("<TargetItemTitle>", ori_samples['TargetItemTitle'][k])
            prompt_ = prompt_.replace("<TargetItemID>", unk_)
            prompt_list.append(prompt_)
        
        self.llama_tokenizer.padding_side = "left"
        prompts_tokens = self.llama_tokenizer(
            prompt_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(ori_samples['UserID'].device)
            
        replaced_idx = torch.nonzero(prompts_tokens.input_ids==self.llama_tokenizer.unk_token_id)

        prompt_embeds = self.llama_model_lora.base_model.model.model.embed_tokens(prompts_tokens.input_ids)
        
        if  "<TargetItemID>" in prompt_ori:
            try:
                prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = samples['TargetItem_emb']
            except:
                print(replaced_idx.shape,'sample',samples['TargetItem_emb'].shape)
                print("#######prmpt decoded example: ",' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))

        return prompt_embeds, prompts_tokens.attention_mask


    def forward(self,samples):
        if self.run_mode_ == 'v1':
            return self.forward_v1(samples)
        elif self.run_mode_ == 'v2':
            return self.forward_v2(samples)
        elif self.run_mode_ == 'v3':
            return self.forward_v3(samples)
        else:
            raise NotImplementedError("None-template version has not been implemtned...")  


    def prompt_based_encode_v2(self,prompt, samples):
        id_orders = get_ids_order(prompt)
        samples_encode, sample_atts = self.encode_recdata_v2(samples,ids_order=id_orders)
        sample_embeds, sample_atts = self.recprompt_wrap_v2(samples_encode, samples, sample_atts, prompt)
        return sample_embeds, sample_atts


    def forward_v2(self, samples):
        user_selective_prompts = False
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_with_p([5,5,5,1])) #[1,5,3,1]  #[2,5,3,1]
            sample_embeds, sample_atts = self.prompt_based_encode_v2(prompt,samples)
        self.llama_tokenizer.padding_side = "right"
        device = samples['UserID'].device #samples_encode['User_emb'].device

        ans_ = {1:self.pos_ans[0], 0:self.neg_ans[0]}

        text = [ans_[int(t)] for t in samples["label"]] 

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([sample_atts.shape[0],sample_atts.shape[1]],dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([sample_atts, to_regress_tokens.attention_mask], dim=1)

        print(f'{inputs_embeds.shape[1]}',file=open('token_lens.txt','a'))
        with self.maybe_autocast():
            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )

        # new loss, just focus on the target pos and neg tokens 
        pos_ans_id = self.llama_tokenizer(ans_[int(1)],add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(ans_[int(0)],add_special_tokens=False).input_ids[0]
        logits = outputs.logits[:,-t_posi,:][:,pos_ans_id]

        # logits = torch.nan_to_num(logits, nan=-100)
        # logits = torch.sigmoid(logits)
        
        loss = nn.functional.binary_cross_entropy_with_logits(logits, samples['label'].float()) #+ 1e-7 * samples_encode['loss_c']
        return {"loss": loss}


    def generate_for_samples_v2(self, samples,return_all=False):
        # sample = samples["image"]
        user_selective_prompts = False
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
            # vqa_prompt = '###Human: <Img><ImageHere></Img> '
            # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            if user_selective_prompts:  # automatically setting prompt according to the prompt_flag
                prompt_flag = samples['prompt_flag']
                unique_flags = torch.unique(prompt_flag)
                sample_embeds = []
                sample_atts = []
                true_idx = torch.zeros_like(prompt_flag)
                pre_ = 0
                for k_flag in unique_flags:
                    idx_k = torch.nonzero(prompt_flag==k_flag)[0]
                    true_idx[idx_k] = pre_ + torch.arange(idx_k.shape[0])
                    pre_ += idx_k.shape[0]
                    sub_k_sample = {}
                    for key_ in samples.keys():
                        sub_k_sample[key_] = samples[key_][idx_k]
                    if k_flag == 0:   # assume the fist prompt does not use ID information, for cold items
                        used_prompt = self.prompt_list[-1]
                    else:
                        used_prompt = self.prompt_list[1] # during inference, use ID+title information by default.
                    sample_embeds_k, sample_atts_k = self.prompt_based_encode_v2(used_prompt, sub_k_sample)
                    sample_embeds.append(sample_embeds_k)
                    sample_atts.append(sample_atts_k)
                sample_embeds = torch.cat(sample_embeds, dim=0)
                sample_atts = torch.cat(sample_atts,dim=0)
                sample_embeds = sample_embeds[true_idx]
                sample_atts = sample_atts[true_idx]
            else:
                prompt = self.prompt_list[0]
                sample_embeds, sample_atts = self.prompt_based_encode_v2(prompt,samples)

        self.llama_tokenizer.padding_side = "right"



        device = samples['UserID'].device #samples_encode['User_emb'].device

        pos_ans = self.pos_ans[0]
        neg_ans = self.neg_ans[0]
        ans_ = {1:pos_ans, 0:neg_ans}

        # text = ["### Response: " + ans_[int(t)]  for t in samples["label"]]
        text = [ ans_[int(t)]  for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([sample_atts.shape[0],sample_atts.shape[1]],dtype=torch.long).to(device).fill_(-100)

        targets = torch.cat([empty_targets, targets], dim=1)

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([sample_atts, to_regress_tokens.attention_mask], dim=1)

        print(f'{inputs_embeds.shape[1]}',file=open('token_lens.txt','a'))
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
                labels=targets,
            )
            if self.use_quest:
                self.llama_model.quest_clear()

        # loss = outputs.loss
        pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(neg_ans, add_special_tokens=False).input_ids[0]

        logits_ = outputs.logits[:,-t_posi,:][:,pos_ans_id]
        # print(lo)
        loss = nn.functional.binary_cross_entropy_with_logits(logits_, samples['label'].float())

        if return_all:
            return outputs, logits_

        return {"loss": loss, 'logits':logits_}


    def encode_recdata_v2(self, sample, ids_order=None):  # used for stage2
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        
        with self.maybe_autocast():
            batch_size = sample['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            if self.rec_model_type == "sasrec":  # for sasrec, there is no user encoder but just seqs encoder, we take it to get user representation
                user_embeds = self.rec_encoder.seq_encoder(sample['sas_seq']).unsqueeze(-2)
            elif self.rec_model_type == "DCN" or self.rec_model_type == "DIN":
                """
                not really user embeding, but the embedding merged for one sample point
                """
                user_embeds = self.rec_encoder.all_encode(sample['UserID'],sample['TargetItemID'],sample['sas_seq'][:,-10:]).unsqueeze(-2)
            else:
                user_embeds = self.rec_encoder.user_encoder(sample['UserID'], all_users=all_user_embeds).unsqueeze(-2)
            # ***Note: here, for sasrec, item embedding comes form the last layer 
            targetItem_embed = self.rec_encoder.item_encoder(sample['TargetItemID'], all_items=all_item_embeds).unsqueeze(-2)
            
            

            user_embeds_llama = self.llama_proj(user_embeds).reshape(batch_size,-1, self.proj_token_num, hidden_size)
            # if self.rec_encoder !="DCN":
            targetItem_embeds_llama = self.llama_proj(targetItem_embed).reshape(batch_size,-1, self.proj_token_num, hidden_size)
            
            # loss_c = consitence_loss(user_embeds, user_embeds_llama) + consitence_loss(targetItem_embed, targetItem_embeds_llama)
            # print('which',len(ids_order),sample.keys())
            if 'InteractedItemIDs_pad' in sample.keys() and len(ids_order)==3:
                interactedItem_embeds = self.rec_encoder.item_encoder(sample['InteractedItemIDs_pad'], all_items=all_item_embeds)
                interactedItem_embeds_llama = self.llama_proj(interactedItem_embeds).reshape(batch_size,-1, self.proj_token_num, hidden_size)

                merged_embeds = [user_embeds_llama, interactedItem_embeds_llama, targetItem_embeds_llama]
                merged_embeds = [merged_embeds[k] for k in ids_order]
                merged_embeds = torch.cat(merged_embeds,dim=1)
                idx_flag = torch.ones_like(sample['InteractedItemIDs_pad'])
                idx_flag = torch.where(sample['InteractedItemIDs_pad']==self.rec_encoder.padding_index, 0, idx_flag) # indx_of_paddded historical items
                # to indicate user_id, his_items_id, target_item_id
                idx_flag = [torch.ones([idx_flag.shape[0],1]).to(idx_flag.device),idx_flag,torch.ones([idx_flag.shape[0],1]).to(idx_flag.device)]
                idx_flag = [idx_flag[k] for k in ids_order]
                idx_flag = torch.cat(idx_flag,dim=1).to(device)
                idx_nopad = torch.nonzero(idx_flag)
                
                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'InteractedItems_embs': interactedItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'merged_embs': merged_embeds[idx_nopad[:,0],idx_nopad[:,1]].reshape(-1, hidden_size),
                    # 'loss_c': loss_c
                }
            else:
                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'InteractedItems_embs': None,
                    'merged_embs': None,
                    # 'loss_c': loss_c
                }
        sample_atts_llama = None
        # {
        #     'user': atts_user,
        #     'TargetItem': atts_targetItem,
        #     'InteractedItems': atts_interactedItem
        # }
        return sample_embeds_llama, sample_atts_llama


    def recprompt_wrap_v2(self, samples, ori_samples, atts_sample, prompt): # used for stage 2
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<UserID>", "<ItemIDList>", "<ItemTitleList>", "<TargetItemID>", "<TargetItemTitle>"]
            batch_size = ori_samples['UserID'].shape[0]
            bos = "<s>"
            unk_ = self.llama_tokenizer.unk_token #"<unk>"
            unk_ = ".".join([unk_]*self.proj_token_num)
            prompt = bos + prompt # add the bos
            prompt = prompt.replace("<UserID>", unk_)
            prompt = prompt.replace("<TargetItemID>", unk_)

            prompt = prompt.replace("<DCNFeature>", unk_)

            # interactedItems = samples['InteractedItemTitles']
            prompt_list = []
            
            
            for k in range(batch_size):
                prompt_ = prompt+""
                # prompt_ = prompt.replace('UserID',unk_)
                # item_num = samples['interacted']
                if 'InteractedNum' in ori_samples.keys():
                    prompt_ = prompt_.replace('<ItemIDList>', ', '.join([unk_]*ori_samples['InteractedNum'][k]))
                    prompt_ = prompt_.replace("<ItemTitleList>", ori_samples['InteractedItemTitles'][k])
                prompt_ = prompt_.replace("<TargetItemTitle>", ori_samples['TargetItemTitle'][k])
                # prompt_ = prompt_.replace("<TargetItemID>", unk_)
                # prompt_ += samples['Response'][k]
                prompt_list.append(prompt_)
            
            if not self.has_print_prompt:
                print("prompt example:", random.choice(prompt_list))
                self.has_print_prompt = True
            
            
            self.llama_tokenizer.padding_side = "right"
            prompts_tokens = self.llama_tokenizer(
            prompt_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(ori_samples['UserID'].device)
            unk_token_id = self.llama_tokenizer.unk_token_id
            if not self.has_pri_decode:
                print("#######prmpt decoded example: ",' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))
                self.has_pri_decode = True
                
            print(f"{max([len(prompt_) for prompt_ in prompt_list])}",file=open('text_lens.txt','a'))
            replaced_idx = torch.nonzero(prompts_tokens.input_ids==unk_token_id)
            if not self.use_lora:
                prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
            else:
                prompt_embeds = self.llama_model_lora.base_model.model.model.embed_tokens(prompts_tokens.input_ids)
            # prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = samples['merged_embs']
            if "<UserID>" in prompt_ori  and "<ItemIDList>" in prompt_ori and  "<TargetItemID>" in prompt_ori:
                prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = samples['merged_embs']
            elif "<UserID>" in prompt_ori and "<TargetItemID>" in prompt_ori and "<ItemIDList>" not in prompt_ori:
                try:
                    prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = torch.cat([samples['User_emb'], samples['TargetItem_emb']],dim=-2).reshape(-1,samples['User_emb'].shape[-1])
                except:
                    print(replaced_idx,'sample',samples['User_emb'].shape,samples['TargetItem_emb'].shape)
            elif "<DCNFeature>" in prompt_ori:
                prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = samples['User_emb'].reshape(-1,samples['User_emb'].shape[-1])
            else:
                pass 
            return prompt_embeds, prompts_tokens.attention_mask


    def encode_recdata_v1(self, sample): # used for stage1
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        with self.maybe_autocast():
            all_user_embeds, all_items_embeds = self.rec_encoder.computer()
            user_embeds = self.rec_encoder.user_encoder(sample['UserID'],all_users=all_user_embeds).unsqueeze(-2)
            targetItem_embed = self.rec_encoder.item_encoder(sample['PairItemIDs'],all_items=all_items_embeds)
            

            user_embeds_llama = self.llama_proj(user_embeds)
            targetItem_embeds_llama = self.llama_proj(targetItem_embed)
        
        sample_embeds_llama = {
            'User_emb': user_embeds_llama,
            'PairItem_emb': targetItem_embeds_llama,
        }
        sample_atts_llama = None
        return sample_embeds_llama, sample_atts_llama

    def recprompt_wrap_v1(self, samples, ori_samples, atts_sample, prompt): # used for stage 1
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<UserID>", "<ItemID>"]
            batch_size = ori_samples['UserID'].shape[0]
            bos = "<s>"
            unk_ = self.llama_tokenizer.unk_token #"<unk>"
            prompt = bos + prompt # add the bos
            prompt = prompt.replace("<UserID>", unk_)
            prompt = prompt.replace("<ItemID>", unk_)
            # interactedItems = samples['InteractedItemTitles']
            prompt_list = []
            
            
            for k in range(batch_size):
                prompt_ = prompt+""
                prompt_list.append(prompt_)
            
            # print(prompt_)
            
            self.llama_tokenizer.padding_side = "right"
            prompts_tokens = self.llama_tokenizer(
            prompt_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(samples['User_emb'].device)
            unk_token_id = self.llama_tokenizer.unk_token_id
            replaced_idx = torch.nonzero(prompts_tokens.input_ids==unk_token_id)
            prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
            if "<UserID>" in prompt_ori and "<ItemID>" in prompt_ori:
                prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]]  = torch.cat([samples['User_emb'], samples['PairItem_emb']],dim=-2).reshape(-1,samples['User_emb'].shape[-1])
            else:
                raise RuntimeError("the pretraining just support one type prompt") 
            return prompt_embeds, prompts_tokens.attention_mask

    def forward_v1(self, samples):
        # sample = samples["image"]
        samples_encode, sample_atts = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            sample_embeds, sample_atts = self.recprompt_wrap_v1(samples_encode, samples, sample_atts, prompt)

        self.llama_tokenizer.padding_side = "right"



        device = samples['UserID'].device #samples_encode['User_emb'].device

        # ans_ = {1: "The user prefers the former item because its MF embedding is a closer match to the user's MF embedding than that of the latter item.", 
        #         0: "The user prefers the latter item because its MF embedding is a closer match to the user's MF embedding than that of the former item."}

        # pos_ans = ["The former item.", "The first item.", "The former.", "The first.", "The former one.", "The first one.",]
        # neg_ans = ["The latter item.", "The second item.", "The latter.", "The second.", "The latter one.", "The second one."]
        # pos_ans = ["The first item."]
        # neg_ans = ["The second item."]
        ans_ = {1: self.pos_ans, 
                0: self.neg_ans}
        text = [random.choice(ans_[int(t)]) + self.end_sym for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([sample_atts.shape[0],sample_atts.shape[1]],dtype=torch.long).to(device).fill_(-100)

        # empty_targets = (
        #     torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
        #                dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        # )
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([sample_atts, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}
    
    
    def generate_for_samples_v1(self, samples):
        
        
        samples_encode, sample_atts = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
            # vqa_prompt = '###Human: <Img><ImageHere></Img> '
            # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            sample_embeds, sample_atts = self.recprompt_wrap_v1(samples_encode, samples, sample_atts, prompt)

        self.llama_tokenizer.padding_side = "right"



        device = samples_encode['User_emb'].device
        # sample = samples["image"]
        # ans_ = {1: "The user prefers the former item because its MF embedding is a closer match to the user's MF embedding than that of the latter item.", 
        #         0: "The user prefers the latter item because its MF embedding is a closer match to the user's MF embedding than that of the former item."}
        # pos_ans = ["The former item.", "The first item.", "The former.", "The first.", "The former one.", "The first one."]
        # neg_ans = ["The latter item.", "The second item.", "The latter.", "The second.", "The latter one.", "The second one."]
        # pos_ans = ["The first item."]
        # neg_ans = ["The second item."]
        # pos_ans = ["The first item."]
        # neg_ans = ["The second item."]
        ans_ = {1: self.pos_ans, 
                0: self.neg_ans}
        text = [random.choice(ans_[int(t)]) + self.end_sym for t in samples["label"]]

        # text = [ans_[int(t)] + self.end_sym for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([sample_atts.shape[0],sample_atts.shape[1]],dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([sample_atts, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        return {"loss": loss}
    

    def generate_for_samples(self,samples):
        if self.run_mode_ == 'v1':
            return self.generate_for_samples_v1(samples)
        elif self.run_mode_ == 'v2':
            return self.generate_for_samples_v2(samples)
        elif self.run_mode_ == 'v3':
            return self.generate_for_samples_v3(samples)
        else:
            raise NotImplementedError("Not implement the default version")     

    def prompt_with_p(self,p):
        if self.prompt_list_p is None:
            prompt_list_p= []
            for k in range(len(p)):
                prompt_list_p.extend([self.prompt_list[k]]*p[k])
            self.prompt_list_p = prompt_list_p
            return self.prompt_list_p
        else:
            return self.prompt_list_p


    @classmethod
    def from_config(cls, cfg):
        # rec_model="MF",
        # embedding_size=64,
        # freeze_rec=True,
        # rec_precision='fp16',
        # rec_config = None,
        # llama_model="",
        # prompt_path="",
        # prompt_template="",
        # max_txt_len=32,
        # end_sym='\n',
        # low_resource=False,  # use 8 bit and put vit in cpu
        # device_8bit=0,  # the device of 8bit 


        rec_model = cfg.get('rec_model',"MF")
        rec_config = cfg.rec_config
        embedding_size = cfg.get("rec_emb_size")
        freeze_rec = cfg.get("freeze_rec",True)
        rec_precision = cfg.get("rec_precision", 'fp16')
        rec_config = cfg.get("rec_config")
        lora_config = cfg.get("lora_config")
        loss_config = cfg.get("loss_config", None)
        quest_config = cfg.get("quest_config", None)
        cake_config = cfg.get("cake_config", None)
        llama_model = cfg.get("llama_model")
        proj_token_num = cfg.get("proj_token_num")
        proj_mid = cfg.get("proj_mid_times")
        freeze_proj = cfg.get("freeze_proj")
        freeze_lora = cfg.get("freeze_lora")
        infer_type = cfg.get("infer_type", "native")


        # drop_path_rate = cfg.get("drop_path_rate", 0)
        # use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        # vit_precision = cfg.get("vit_precision", "fp16")
        # freeze_vit = cfg.get("freeze_vit", True)
        # freeze_qformer = cfg.get("freeze_qformer", True)


        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')


        model = cls(
            rec_model=rec_model,
            rec_config=rec_config,
            pretrained_rec = rec_config['pretrained_path'],
            freeze_rec=freeze_rec,
            rec_precision=rec_precision,
            infer_type=infer_type,
            quest_config=quest_config,
            cake_config=cake_config,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            proj_token_num = cfg.get("proj_token_num"),
            proj_drop = cfg.get("proj_drop"),
            lora_config = lora_config,
            loss_config = loss_config,
            proj_mid = proj_mid,
            freeze_lora=freeze_lora,
            freeze_proj=freeze_proj
        )

        generate_config = cfg.get("generate_config", None)
        ckpt_path = cfg.get("ckpt", None)  # load weights of MiniGPT-4
        user2group = cfg.get('user2group',None)
        model.set_generate_config(generate_config,ckpt_path)
        if isinstance(ckpt_path,str):
            print("Load MiniGPT4Rec Checkpoint: {}".format(ckpt_path))
            #是文件吗
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                # msg = model.load_state_dict(ckpt['model'], strict=False)
                msg = model.load_state_dict(ckpt['model'], strict=False)
                print("loading message, msg.... ", msg)
            else: #eval
                tag = ckpt_path.split('/')[-1]
                ckpt_path = '/'.join(ckpt_path.split('/')[:-1])
                # models_name = os.listdir(ckpt_path)
                model.load_state_dict(torch.load(os.path.join(ckpt_path,f"project_model_{tag}.pth"), map_location="cpu"), strict=False)
                adapters = os.listdir(os.path.join(ckpt_path,"lora_adapter"))
                for adapter in adapters:
                    model.llama_model_lora.load_adapter(os.path.join(ckpt_path,"lora_adapter",adapter), adapter_name=adapter)
                model.llama_model_lora.set_adapter("group0")
                if len(adapters)>1:
                    model.set_user2group(user2group)
                print("loading adapters and proj")
        elif ckpt_path is not None:#stage2,一个也可以
            print('loading adapters')
            for idx,adapter in enumerate(ckpt_path):
                model.llama_model_lora.load_adapter(adapter, adapter_name=f"group{idx}")
            model.llama_model_lora.set_adapter("group0")
            if len(ckpt_path)>1:
                model.set_user2group(user2group)

        # reload the rec model, avoiding it be covered by the loaded ckpt
        if os.path.exists(rec_config['pretrained_path']) and freeze_rec:
            model.rec_encoder.load_state_dict(torch.load(rec_config['pretrained_path'], map_location="cpu"))
        ans_type = cfg.get('ans_type')
        
        model.set_answer_type(mode=ans_type)
        model.print_prompt()
        return model
