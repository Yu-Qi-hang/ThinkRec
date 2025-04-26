gpu=$1
port=$2
stragety=$3
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
mfdir=/data/yuqihang/result/CoLLM/checkpoints/mf
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

if [[ $stragety == *'tallrec'* ]]; then
use_ids=False
use_desc=False
prompt_path=prompts/tallrec_amazon_.txt
generate_config=False
model_dir='['$6/adapter_auc_uauc']'
elif [[ $stragety == *'collm'* ]]; then
use_ids=False
use_desc=False
prompt_path=prompts/collm_amazon_.txt
generate_config=False
model_dir=$6/auc_uauc
elif [[ $stragety == *'both'* ]]; then
use_ids=True
use_desc=True
prompt_path=prompts/collm_amazon_.txt
generate_config=False
model_dir=$6/auc_uauc
elif [[ $stragety == *'reason'* ]]; then
use_ids=True
use_desc=True
prompt_path=prompts/reflection_amazon.txt
generate_config=True
model_dir=$6/auc_uauc
fi

if [[ $stage == 'eval' ]];then
testdata=\[test_small\]
max_len=16
batch_size_eval=4
eval_text=False
elif [[ $stage == 'test' ]];then
testdata=\[test_tiny\]
generate_config=True
max_len=512
batch_size_eval=2
eval_text=True
fi
echo $testdata $data $stragety $model_dir
CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=$port train_collm_mf_din.py  --cfg-path train_configs/new/reason_mf_eval.yaml --options model.generate_config.enable $generate_config model.generate_config.max_len $max_len model.prompt_path $prompt_path model.rec_config.pretrained_path $mf_model model.ckpt $model_dir datasets.amazon_ood.path $data/ datasets.amazon_ood.build_info.use_ids $use_ids datasets.amazon_ood.build_info.use_desc $use_desc datasets.amazon_ood.build_info.storage $data/ run.test_splits $testdata run.batch_size_eval $batch_size_eval run.eval_text $eval_text datasets.amazon_ood.build_info.user2group $user2group