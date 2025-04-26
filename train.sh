gpu=$1
port=$2
stragety=$3
stage=$4
data=$5
# ckpt=$6
# group=$7

cfg_file=/home/yuqihang/projects/CoLLM/train_configs/new/${stragety}_mf_${stage}.yaml
mfdir=/data/yuqihang/result/CoLLM/checkpoints/mf
if [[ $data == *'yelp'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep yelp|grep '04')
    output_dir=/data/yuqihang/result/CoLLM/reproduce/yelp/
elif [[ $data == *'toy'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep toy|grep '04')
    output_dir=/data/yuqihang/result/CoLLM/reproduce/toys/
elif [[ $data == *'ml1m'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep ml1m|grep '04')
    output_dir=/data/yuqihang/result/CoLLM/reproduce/ml1m/
elif [[ $data == *'beauty'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep beauty|grep '04')
    output_dir=/data/yuqihang/result/CoLLM/reproduce/beauty/
elif [[ $data == *'book'* ]]; then
    mf_model=$(ls ${mfdir}/*.pth|grep book|grep '04')
    output_dir=/data/yuqihang/result/CoLLM/reproduce/book/
fi

if [[ $stragety == *'soft'* ]];then
    mf_model=not_have
fi

if [[ $stage == 'stage1' ]];then
CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=$port train_collm_mf_din.py  \
--cfg-path train_configs/new/${stragety}_mf_${stage}.yaml \
--options model.rec_config.pretrained_path $mf_model datasets.amazon_ood.path $data/ datasets.amazon_ood.build_info.storage $data/ run.output_dir $output_dir

elif [[ $stage == 'stage2' ]];then
ckpt='['$6']'
if [[ $stragety == *'soft'* ]];then
    ckpt=$6/auc_uauc
fi
CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=$port train_collm_mf_din.py  \
--cfg-path train_configs/new/${stragety}_mf_${stage}.yaml \
--options model.rec_config.pretrained_path $mf_model model.ckpt $ckpt datasets.amazon_ood.path $data/ datasets.amazon_ood.build_info.storage $data/ run.output_dir $output_dir

else #stage3
ckpt='['$6']'
CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=$port train_collm_mf_din.py  \
--cfg-path train_configs/new/${stragety}_mf_${stage}.yaml \
--options model.freeze_lora.layers 8 model.ckpt $ckpt datasets.amazon_ood.path $data/ datasets.amazon_ood.build_info.storage $7/ run.output_dir $output_dir  model.rec_config.pretrained_path $mf_model
fi