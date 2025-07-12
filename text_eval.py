import os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from collections import Counter
from bleurt import score as bleurt_score
import torch
import argparse

def tokenize(text, lang="en"):
    """分词函数（支持中英文）"""
    return text.split()

def read_file(file_path):
    """读取文本文件（每行一条数据）"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def calc_bleu(ref, gen, lang="en"):
    """计算BLEU-4分数"""
    ref_tokens = [tokenize(ref, lang)]
    gen_tokens = tokenize(gen, lang)
    smooth = SmoothingFunction().method1  # 处理短文本平滑
    return sentence_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

def calc_rouge(ref, gen, lang="en"):
    """计算ROUGE-L分数"""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=(lang == "en"))
    scores = scorer.score(ref, gen)
    return scores["rougeL"].fmeasure

def calc_meteor(ref, gen):
    """计算METEOR分数（需预下载nltk数据）"""
    # 首次运行需执行: nltk.download('wordnet')
    from nltk.translate.meteor_score import meteor_score
    return meteor_score([ref.split()], gen.split())

def calc_bert_score(refs, gens, lang="en"):
    """计算BERTScore (P/R/F)"""
    P, R, F1 = bert_score(gens, refs, lang=lang)
    return F1.mean().item()

# ========================= 主流程 =========================
def evaluate_gen_quality(ref_file, gen_file, lang="en"):
    """主评估函数"""
    # 读取数据
    if isinstance(ref_file,list):
        refs = ref_file
    else:
        assert os.path.exists(ref_file), f"参考文件{ref_file}不存在！"
        refs = read_file(ref_file)
    if isinstance(gen_file,list):
        gens = gen_file
    else:
        assert os.path.exists(gen_file), f"生成文件{gen_file}不存在！"
        gens = read_file(gen_file)
    assert len(refs) == len(gens), "文件行数不匹配！"
    
    # 初始化结果
    metrics = {
        "BLEU": [], "ROUGE-L": [], "METEOR": []
    }
    
    # 逐条计算指标
    for ref, gen in zip(refs, gens):
        metrics["BLEU"].append(calc_bleu(ref, gen, lang))
        metrics["ROUGE-L"].append(calc_rouge(ref, gen, lang))
        metrics["METEOR"].append(calc_meteor(ref, gen))
    
    # 计算BERTScore（批量计算更高效）
    bert_f1 = calc_bert_score(refs, gens, lang="zh" if lang == "zh" else "en")
    metrics["BERTScore"] = bert_f1
    checkpoint = "/data/yuqihang/model/BLEURT-20"
    scorer = bleurt_score.BleurtScorer(checkpoint)
    scores = scorer.score(references=refs, candidates=gens)
    metrics["BLEURT"] = scores
    # 汇总结果
    results = {}
    for k, v in metrics.items():
        if k != "BERTScore":
            results[f"{k}"] = np.mean(v)
            # results[f"{k}_std"] = np.std(v)
        else:
            results[k] = v
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--ref_file", type=str, default="")
    parser.add_argument("-g","--gen_file", type=str, default="")
    args = parser.parse_args()
    results = evaluate_gen_quality(args.ref_file, args.gen_file, lang="en")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    # dataset = ['book','yelp','ml1m']
    # method = ['prompt4nr','collm_stage1','collm_stage2','reason_stage2']
    # for i in range(len(dataset)):
    #     for j in range(len(method)):
    #         gen_file = f'/data/yuqihang/result/CoLLM/reproduce/{dataset[i]}/{method[j]}/gen.txt'
    #         if j==3:
    #             ref_file = f'/data/yuqihang/datasets/collm-datasets/{dataset[i]}du/reason/test.txt'
    #         elif i==0:
    #             ref_file = f'/data/yuqihang/datasets/collm-datasets/{dataset[i]}new/reason/test.txt'
    #         else:
    #             ref_file = f'/data/yuqihang/datasets/collm-datasets/{dataset[i]}/reason/test.txt'
    #         results = evaluate_gen_quality(ref_file, gen_file, lang="en")
    #         print("评估结果:",dataset[i],method[j])
    #         for k, v in results.items():
    #             print(f"{k}: {v:.4f}")


