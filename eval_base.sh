gpu=$1
data=$2
model=$3

if [[ $model == *'mf'* ]]; then
mdir=/data/yuqihang/result/CoLLM/checkpoints/mf
model=mf
elif [[ $model == *'l'* ]]; then
mdir=/data/yuqihang/result/CoLLM/checkpoints/lightgcn
model=lightgcn
elif [[ $model == *'s'* ]]; then
mdir=/data/yuqihang/result/CoLLM/checkpoints/sasrec
model=sasrec
fi

if [[ $data == *'yelp'* ]]; then
    mf_model=$(ls ${mdir}/*.pth|grep yelp|grep '04')
    dataset=/data/yuqihang/datasets/collm-datasets/yelp
elif [[ $data == *'toy'* ]]; then
    mf_model=$(ls ${mdir}/*.pth|grep toy|grep '04')
    dataset=/data/yuqihang/datasets/collm-datasets/toys
elif [[ $data == *'ml1m'* ]]; then
    mf_model=$(ls ${mdir}/*.pth|grep ml1m|grep '04')
    dataset=/data/yuqihang/datasets/collm-datasets/ml1m
elif [[ $data == *'beauty'* ]]; then
    mf_model=$(ls ${mdir}/*.pth|grep beauty|grep '04')
    dataset=/data/yuqihang/datasets/collm-datasets/beauty
else
    mf_model=$(ls ${mdir}/*.pth|grep book|grep '04')
    dataset=/data/yuqihang/datasets/collm-datasets/booknew
fi

CUDA_VISIBLE_DEVICES=$gpu python baseline_train_$model.py --data_dir $dataset --save_path $mf_model