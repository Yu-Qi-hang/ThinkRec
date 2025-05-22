# ThinkRec: Thinking-based recommendation via LLM

## 📁 Code Structure
```shell
ThinkRec
├── minigpt4: Core code of ThinkRec, following the structure of CoLLM.
│   ├── models: Defines our ThinkRec model architecture.
│   ├── datasets: Defines dataset classes.
│   ├── task: A overall task class, defining the used model and datasets, training epoch and evaluation.
│   ├── runners: A runner class to train and evaluate a model based on a task.
│   └── common: Commonly used functions.
├── dataset: Datasets processing files.
│   ├── tools: General tools for dataset preprocessing.
│   └── {dataset_name}: Files for preprocessing specific dataset.
├── prompt: Used prompts.
├── train_configs: Training configuration files, setting hyperparameters.
├── train_collm_xx.py ThinkRec training file.
├── train_xx.sh ThinkRec training scripts.
├── baseline_train_xx.py: Baseline training file.
├── user_group.py dataset grouping file.
├── text_eval.py Evaluation of the generated text.
└── eval_xx.py Evaluation scripts for performance.
```
## ⚙️ Environment Setup

```shell
conda create -n thinkrec python=3.9.20
conda activate thinkrec
python -m pip install --upgrade pip==24.2
pip install torchvision==0.20.0 torchaudio==2.5.0
pip install -r requirements.txt
pip install torch==2.5.0
```
There will be some version conflicts, you can install it in stages, and reinstall the corresponding version of the package if it is overwritten.

## 🗃️ Dataset & Model Preparation
download raw datasets from [Amazon2018](https://cseweb.ucsd.edu/∼jmcauley/datasets/amazon_v2/), [Yelp](https://business.yelp.com/data/resources/open-dataset/), [MoveiLens-1M](https://grouplens.org/datasets/movielens/1m/).
Refer to the dataset folder to process your own data
Download the pretrained Llama3-8B model from [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and add `<unk>` as unknown token as follows:
| file | add or modify |
|--------|--------|
| config.json | "unk_token_id": 128002 |
| generation_config.json | "unk_token_id": 128002 |
| special_tokens_map.json | "unk_token": "<unk>" |
| tokenizer_config.json | "unk_token": "<unk>" |
| tokenizer.json | "id": 128002,"content": "<unk>" |

## 🚀 Model Training
### ⚡ small model
```shell
python baseline_train_***.py --istrain --data_dir /path/to/datadir/ --save_path /path/to/savedir/
```

### 💡 ThinkRec

```shell
#stage1 
bash train_mf.sh 0 11119 reason stage1 /path/to/datadir/
#stage2
bash train_mf.sh 0 11119 reason stage2 /path/to/datadir/ /path/to/stage1/checkpoint/
#stage3 
python user_group.py --data_dir /path/to/datadir/ --pretrained_rec /path/to/small/model/ --mode h --n 2 # grouped datasets
bash train_mf.sh 0 11119 reason stage3 /path/to/datadir/ /path/to/stage1/checkpoint/ /path/to/grouped/datadir/
```
Other methods are similar. Configs are in `/train_configs/new`

## 📊 Evaluation
```shell
bash eval_reason.sh 0 11119 mf eval /path/to/datadir/ /path/to/checkpoint/
```
To calculate the quality of reasons, you should install [Bleurt](https://github.com/google-research/bleurt) and run `test_eval.py`

## 🔐 License
This repository is under  [BSD 3-Clause License](./LICENSE). Many codes are based on [CoLLM](https://github.com/zyang1580/CoLLM) with [BSD 3-Clause License](./LICENSE_Collm.md), which is build upon [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) with [BSD 3-Clause License](./LICENSE_minigpt4.md) and [Lavis](https://github.com/salesforce/LAVIS) with [BSD 3-Clause License](./LICENSE_Lavis.md).
