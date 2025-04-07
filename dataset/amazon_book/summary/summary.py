import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

def get_desc(meta:dict)->str:
    description = ''
    candidates = ['category','brand','price','description']
    for item in candidates:
        if item in meta:
            sentence = meta[item].replace("...",".").replace("&","and")
            dirty = ['visit','amazon','visit amazon','description','price','brand','category','page','book','books','author','publisher']
            for word in dirty:
                sentence = sentence.replace(word,'')
            description = f'{description} {" ".join(sentence.split()[:100])}.'
    return description

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("damo/nlp_polylm_qwen_7b_text_generation", revision = 'v1.0.1', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("damo/nlp_polylm_qwen_7b_text_generation", revision = 'v1.0.1',trust_remote_code=True, bf16=True).eval().to(device)
    
    print("Configing generation...")
    # 创建 GenerationConfig 对象
    generation_config = GenerationConfig.from_pretrained(
        "damo/nlp_polylm_qwen_7b_text_generation",
        revision='v1.0.1',
        trust_remote_code=True
    )
    # 更新配置参数
    generation_config.max_new_tokens = 32
    generation_config.min_length = 1
    generation_config.do_sample = False  # 禁用采样，启用贪婪搜索
    generation_config.num_beams = 2      # 使用束搜索，宽度为 4
    generation_config.num_return_sequences = 1  # 返回一个序列
    generation_config.repetition_penalty = 1.2  # 重复惩罚系数
    generation_config.no_repeat_ngram_size = 2  # 不重复的 n-gram 大小
    generation_config.early_stopping = True
    # 将配置应用到模型
    model.generation_config = generation_config

    json_path = '/home/yuqihang/projects/CoLLM/collm-datasets/booknew/id2title.json'
    save_path = '/home/yuqihang/projects/CoLLM/collm-datasets/booknew/id2keywords2.json'
    rawdict = json.load(open(json_path, 'r'))

    data_types = ['book','movie']
    data_type_id = 0
    prompt_prefix = f'You are a skilled text summarizer. Your task is to extract up to ten key words from the given profile of the {data_types[data_type_id]} above. Answers should contain only keywords, which should be separated by commas.\nKeywords:'

    print("Start summarizing...")
    newdict = {}
    sbar = tqdm(total=len(rawdict))
    cnt = 0
    for idx,meta in rawdict.items():
        cnt += 1
        inputs = tokenizer(f'{get_desc(meta)}\n{prompt_prefix}:', return_tensors='pt')
        inputs = inputs.to(device)
        pred = model.generate(**inputs)
        raw = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)

        raw_words = raw.split('Keywords:')[-1].strip()
        cleaned_words = raw_words.replace('"','').replace(':','')
        cleaned_words = re.sub(r'\s+', ' ', cleaned_words)
        cleaned_words = re.split(r'\s*,\s*', cleaned_words.strip())
        filter_words = list(filter(lambda x: x.lower() not in ['book','books','summary','summarize',''], cleaned_words))[:10]
        keywords = ', '.join(filter_words)

        newdict[idx] = {"title": rawdict[idx]["title"], "keywords":keywords}
        if cnt % 1000 == 0:
            json.dump(newdict, open(save_path, "w"), indent=4)
        sbar.set_postfix(keywords=keywords)
        sbar.update()

    json.dump(newdict, open(save_path, "w"), indent=4)