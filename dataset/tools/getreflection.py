import os
import re
import argparse
import random
import time

import numpy as np
import pandas as pd
import openai
from tqdm import tqdm
from openai import OpenAI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wrong_file', type=str, default='/home/yuqihang/projects/CoLLM/collm-datasets/bookdu/reflection/wrong_1.txt')
    parser.add_argument('--api_key', nargs="+", type=str, default=['Your API Key'])
    args = parser.parse_args()
    print('loading wrong data...')
    wrong_file = args.wrong_file
    if 'toy' in wrong_file:
        data_types = 'toys or games'
        data_type = 'toy or game'
    elif 'ml1m' in wrong_file:
        data_types = 'movies'
        data_type = 'movie'
    elif 'beauty' in wrong_file:
        data_types = 'beauty'
        data_type = 'beauty'
    elif 'yelp' in wrong_file:
        data_types = 'businesses'
        data_type = 'business'
    else:
        data_types = 'books'
        data_type = 'book'
    with open(wrong_file,'r') as f:
        wrong_list = f.readlines()
    print(f'wrong data num:{len(wrong_list)}')
    uid = []
    iid = []
    his = [] 
    label = []
    his_label = []
    his_title = []
    title = []
    for line in wrong_list:
        data = line.split('\sep')
        uid.append(int(data[0]))
        iid.append(int(data[1]))
        his.append(eval(data[2]))
        label.append(int(data[3]))
        his_label.append(eval(data[4]))
        his_title.append(data[5])
        title.append(data[6].replace('\n',''))

    wrong_data = pd.DataFrame({"uid":uid,'iid':iid,'label':label, 'his':his,'his_label':his_label,'his_title':his_title,'title':title})

    wrong_data['his_pad'] = wrong_data['his'].map(lambda x: [0]+x if len(x) < 10 else x)
    wrong_data['his_label_pad'] = wrong_data['his_label'].map(lambda x: [0]+x if len(x) < 10 else x)

    wrong_data = wrong_data[['uid','iid','label','his_pad','his_label_pad','his_title','title']]
    wrong_data.columns = ['uid','iid','label','his','his_label','his_title','title']
    print(f'data view:\n{wrong_data.head()}')

    # wrong_data.to_csv(wrong_file.replace('.txt','.csv'),index=False)
    # wrong_data = pd.read_csv('/home/yuqihang/projects/CoLLM/collm-datasets/bookdu/reflection/wrong_1.csv')

    url = 'https://api.siliconflow.cn/v1/'
    api_key = args.api_key
    def get_response(client_id,message,max_tokens):
        for delay_secs in (2**x for x in range(0, 10)):
            try:
                client = OpenAI(base_url=url,api_key=api_key[client_id%len(api_key)])
                response = client.chat.completions.create(
                    model="Qwen/QwQ-32B",
                    messages=message,
                    stream=False, 
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                reasoning_content = response.choices[0].message.reasoning_content
                return content, reasoning_content
            except openai.OpenAIError as e:
                randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                sleep_dur = delay_secs + randomness_collision_avoidance
                client_id += 1
                print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                time.sleep(sleep_dur)
                continue
        return None, None

    label2text = {0:'No',1:'Yes'}
    prompt_template = '''A user has given high ratings to the following books: <HisItemList>. Using all available information, make a prediction about whether the user would enjoy the book titled <TargetItemTitle>?'''
    hints = [
        'The correct response is <answer>. Reflect on multiple aspects based on historical information and explain the reason for the oversight based on the previous analysis. Reanalyze to make a prediction about whether the user would enjoy the book titled <TargetItemTitle>?',
        'The accurate answer is <answer>. Delve into various aspects considering historical data, elucidate the cause of the oversight according to the preceding analysis. Conduct a reanalysis to forecast whether the user will take pleasure in the book named <TargetItemTitle>?',
        'The precise reply is <answer>. Examine multiple dimensions in light of historical information, and clarify the reason for the oversight based on the prior analysis. Reassess to make a prediction regarding whether the user would relish the book titled <TargetItemTitle>?',
        'The correct answer is <answer>. Consider several aspects in conjunction with historical information, and explain the rationale for the oversight as per the previous analysis. Reanalyze to predict if the user will enjoy the book called <TargetItemTitle>?',
        'The right response is <answer>. Reflect on a variety of aspects with reference to historical information, and account for the oversight based on the earlier analysis. Reanalyze to determine whether the user would appreciate the book titled <TargetItemTitle>?',
        'The correct answer is <answer>. Look into different aspects in the context of historical information, and explain the cause of the oversight in light of the previous analysis. Reanalyze to make a prediction about whether the user would be fond of the book titled <TargetItemTitle>?'
    ]
    
    batch_size = 50
    total_batches = (len(wrong_data) + batch_size - 1) // batch_size  # 计算总批次

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(wrong_data))
        batch_data = wrong_data.iloc[start_idx:end_idx]
        reasons = []
        total_cnt = 0
        sbar = tqdm(total=len(batch_data))
        sbar.set_description(f'Generating Reasoning for Batch {batch_idx+1}/{total_batches}')
        for idx in range(len(batch_data)):
            max_tokens = 1024
            row = batch_data.iloc[idx]
            label = row['label']
            his_seq = row['his_title']
            target = row['title']
            message = prompt_template.replace('book',data_type).replace('books',data_types).replace('<HisItemList>', his_seq).replace('<TargetItemTitle>', target)
            his_message = [
                {"role": "system", "content": "You are a helpful recommender answer with Yes or No."},
                {"role": "user", "content": message}
            ]
            answer, reasoning_content = get_response(idx % len(api_key), his_message, max_tokens)
            cnt = 1
            while label2text[label] not in answer:
                if reasoning_content is None:
                    reasoning_content = ""
                    break
                if len(answer) == 0 and max_tokens < 3000:
                    max_tokens = 512 + max_tokens
                elif len(his_message) < 4:
                    his_message.append({"role": "assistant", "content": label2text[1 - label]})
                    his_message.append({"role": "user", "content": hints[0].replace('book',data_type).replace('books',data_types).replace('<answer>', label2text[label]).replace('<TargetItemTitle>', target)})
                else:
                    his_message.pop()
                    his_message.append({"role": "user", "content": hints[cnt % 6].replace('book',data_type).replace('books',data_types).replace('<answer>', label2text[label]).replace('<TargetItemTitle>', target)})
                answer, reasoning_content = get_response(idx % len(api_key), his_message, max_tokens)
                cnt += 1
            reflect = re.sub(r'\s+', ' ', reasoning_content.strip())
            reasons.append(f"{label2text[label]}. {reflect}")
            sbar.set_postfix(cnt=cnt, max_tokens=max_tokens)
            if cnt > 5:
                print(f'cnt:{cnt}')
            total_cnt += cnt
            sbar.update(1)
        sbar.close()

        batch_data['reason'] = reasons
        batch_file = wrong_file.replace('.txt', f'_batch_{batch_idx+1}.csv')
        batch_data.to_csv(batch_file, index=False)
        print(f"Cost {total_cnt} turns, saving batch {batch_idx+1} at {batch_file}")