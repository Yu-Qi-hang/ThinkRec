gpu=$1
port=$2
stragety=$3 #mf lightgcn sasrec
stage=$4
data=$5
model_dir=$6
# user2group=$7
if [ -z "$7" ];then
    user2group="none"
else
    user2group=$7
fi
echo $user2group
mfdir=/data/yuqihang/result/CoLLM/checkpoints/$stragety
if [[ $data == *'yelp'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep yelp|grep '04')
elif [[ $data == *'toy'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep toy|grep '04')
elif [[ $data == *'ml1m'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep ml1m|grep '04')
elif [[ $data == *'beauty'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep beauty|grep '04')
else
    mf_model=$(ls ${mfdir}/*.pth|grep book|grep '04')
fi

use_ids=True
use_desc=True
prompt_path=prompts/reflection_amazon.txt
generate_config=True
model_dir=$6/auc_uauc

if [[ $stage == 'eval' ]];then
testdata=\[test_small\]
max_len=16
batch_size_eval=2
eval_text=False
elif [[ $stage == 'test' ]];then
testdata=\[test_tiny\]
max_len=512
batch_size_eval=2
eval_text=True
fi

if [[ $stragety == *'mf'* ]];then
py_file=train_collm_mf_din.py
yaml_file=train_configs/new/reason_mf_eval.yaml
elif [[ $stragety == *'sasrec'* ]];then
py_file=train_collm_sasrec.py
yaml_file=train_configs/new/reason_sasrec_eval.yaml
elif [[ $stragety == *'lightgcn'* ]];then
py_file=train_collm_lgcn.py
yaml_file=train_configs/new/reason_lightgcn_eval.yaml
fi


echo $testdata $data $stragety $model_dir $mf_model
CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=$port $py_file  --cfg-path $yaml_file --options model.generate_config.enable $generate_config model.generate_config.max_len $max_len model.prompt_path $prompt_path model.rec_config.pretrained_path $mf_model model.ckpt $model_dir datasets.amazon_ood.path $data/ datasets.amazon_ood.build_info.use_ids $use_ids datasets.amazon_ood.build_info.use_desc $use_desc datasets.amazon_ood.build_info.storage $data/ run.test_splits $testdata run.batch_size_eval $batch_size_eval run.eval_text $eval_text datasets.amazon_ood.build_info.user2group $user2group