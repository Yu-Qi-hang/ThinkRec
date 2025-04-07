import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import random
import time

api_key = ['sk-bvzykhzeyxtiwysnwkdnfgccudpytjjxxndypbemdykqdwad','sk-yhlekcjeoqlqajykctfhgvegkfsffutnfcogetfxpizdlgvo']
url = 'https://api.siliconflow.cn/v1/'

def get_desc(meta:dict)->str:
    description = ''
    candidates = ['category','brand','price','title','description']
    for item in candidates:
        if item in meta:
            sentence = meta[item].replace("...",".").replace("&","and")
            dirty = ['visit','amazon','visit amazon','description','price','brand','category','page','book','books','author','publisher']
            for word in dirty:
                sentence = sentence.replace(word,'')
                sentence = sentence.replace(word.capitalize(),'')
            sentence = re.sub(r'\s+', ' ', sentence)
            description = f'{description} {" ".join(sentence.split()[:200])}.'
    return description
def get_response(message):
    client = OpenAI(base_url=url,api_key=random.choice(api_key))
    for delay_secs in (2**x for x in range(0, 10)):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-14B-Instruct",
                messages=message,
                stream=False, 
                max_tokens=16
            )
            content = response.choices[0].message.content
            return content
        except openai.OpenAIError as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + randomness_collision_avoidance
            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            time.sleep(sleep_dur)
            continue

if __name__ == '__main__':
    json_path = '/home/yuqihang/projects/CoLLM/collm-datasets/booknew/id2title.json'
    save_path = '/home/yuqihang/projects/CoLLM/collm-datasets/booknew/id2keywords2.json'
    rawdict = json.load(open(json_path, 'r'))

    data_types = ['book','movie']
    data_type_id = 0

    print("Start summarizing...")
    newdict = rawdict.copy()
    sbar = tqdm(total=len(rawdict))
    cnt = 0
    for idx,meta in rawdict.items():
        cnt += 1
        message = [
            {"role": "system", "content": f"You are a skilled text summarizer. Your task is to extract up to ten key words from the given profile of the {data_types[data_type_id]}. Answers should contain only keywords, which should be separated by commas."},
            {"role": "user", "content": get_desc(meta)}
        ]
        raw_words = get_response(message)
        cleaned_words = raw_words.replace('"','').replace(':','')
        cleaned_words = re.sub(r'\s+', ' ', cleaned_words)
        cleaned_words = re.split(r'\s*,\s*', cleaned_words.strip())
        filter_words = list(filter(lambda x: x.lower() not in ['book','books','summary','summarize',''], cleaned_words))[:10]
        keywords = ', '.join(filter_words)

        newdict[idx]["keywords"] = keywords
        if cnt % 1000 == 0:
            json.dump(newdict, open(save_path, "w"), indent=4)
        sbar.set_postfix(keywords=keywords)
        sbar.update()

    json.dump(newdict, open(save_path, "w"), indent=4)