gpu=$1
port=$2
stragety=$3
stage=$4
data=$5
# ckpt=$6
# group=$7

cfg_file=train_configs/new/${stragety}_lightgcn_${stage}.yaml
mfdir=/data/yuqihang/result/CoLLM/checkpoints/lightgcn
if [[ $data == *'yelp'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep yelp|grep '04')
    output_dir=/data/yuqihang/result/CoLLM/reproduce/yelp/
    dataset=yelp
elif [[ $data == *'toy'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep toy|grep '04')
    output_dir=/data/yuqihang/result/CoLLM/reproduce/toys/
    dataset=toy
elif [[ $data == *'ml1m'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep ml1m|grep '04')
    output_dir=/data/yuqihang/result/CoLLM/reproduce/ml1m/
    dataset=ml1m
elif [[ $data == *'beauty'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep beauty|grep '04')
    output_dir=/data/yuqihang/result/CoLLM/reproduce/beauty/
    dataset=beauty
elif [[ $data == *'book'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep book|grep '04')
    output_dir=/data/yuqihang/result/CoLLM/reproduce/book/
    dataset=book
fi

if [[ $stage == 'stage1' ]];then
CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=$port train_collm_lgcn.py  --cfg-path $cfg_file \
--options model.rec_config.pretrained_path $mf_model datasets.amazon_ood.path $data/ datasets.amazon_ood.build_info.storage $data/ run.output_dir $output_dir model.rec_config.dataset $dataset

elif [[ $stage == 'stage2' ]];then
CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=$port train_collm_lgcn.py  --cfg-path $cfg_file \
--options model.rec_config.pretrained_path $mf_model model.ckpt '['$6']' datasets.amazon_ood.path $data/ datasets.amazon_ood.build_info.storage $data/ run.output_dir $output_dir model.rec_config.dataset $dataset

else #stage3
CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=$port train_collm_lgcn.py  --cfg-path $cfg_file \
--options model.freeze_lora.layers 8 model.ckpt '['$6']' datasets.amazon_ood.path $data/ datasets.amazon_ood.build_info.storage $7/ run.output_dir $output_dir  model.rec_config.pretrained_path $mf_model model.rec_config.dataset $dataset
fi