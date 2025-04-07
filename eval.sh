echo "generate configuration yaml";
python replace_yaml.py --ckptdir $2;

echo "auc eval";
ln -s $2/checkpoint_bestauc.pth $2/checkpoint_best.pth ;
echo "Reflect" >> $2/checkpoint_bestauc.txt;
CUDA_VISIBLE_DEVICES=$1 WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=11189 train_collm_mf_din.py  --cfg-path train_configs/myconfig/reason_mf_book_eval1.yaml >> $2/checkpoint_bestauc.txt 2>&1 ;
cat $2/checkpoint_best.txt >> $2/checkpoint_bestauc.txt;
rm $2/checkpoint_best.txt;
python /home/yuqihang/projects/tools/Mailbox.py --message auc reflect;
echo "Forward" >> $2/checkpoint_bestauc.txt;
CUDA_VISIBLE_DEVICES=$1 WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=11189 train_collm_mf_din.py  --cfg-path train_configs/myconfig/reason_mf_book_eval2.yaml >> $2/checkpoint_bestauc.txt 2>&1;
rm $2/checkpoint_best.pth;
python /home/yuqihang/projects/tools/Mailbox.py --message auc forward;

echo "both eval"
ln -s $2/checkpoint_bestauc_uauc.pth $2/checkpoint_best.pth ;
echo "Reflect" >> $2/checkpoint_bestauc_uauc.txt;
CUDA_VISIBLE_DEVICES=$1 WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=11189 train_collm_mf_din.py  --cfg-path train_configs/myconfig/reason_mf_book_eval1.yaml >> $2/checkpoint_bestauc_uauc.txt 2>&1;
cat $2/checkpoint_best.txt >> $2/checkpoint_bestauc_uauc.txt;
rm $2/checkpoint_best.txt;
python /home/yuqihang/projects/tools/Mailbox.py --message both reflect;
echo "Forward" >> $2/checkpoint_bestauc_uauc.txt;
CUDA_VISIBLE_DEVICES=$1 WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=11189 train_collm_mf_din.py  --cfg-path train_configs/myconfig/reason_mf_book_eval2.yaml >> $2/checkpoint_bestauc_uauc.txt 2>&1;
rm $2/checkpoint_best.pth;
python /home/yuqihang/projects/tools/Mailbox.py --message both forward;

echo "uauc eval"
ln -s $2/checkpoint_bestuauc.pth $2/checkpoint_best.pth ;
echo "Reflect" >> $2/checkpoint_bestuauc.txt;
CUDA_VISIBLE_DEVICES=$1 WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=11189 train_collm_mf_din.py  --cfg-path train_configs/myconfig/reason_mf_book_eval1.yaml >> $2/checkpoint_bestuauc.txt 2>&1;
cat $2/checkpoint_best.txt >> $2/checkpoint_bestuauc.txt;
rm $2/checkpoint_best.txt;
python /home/yuqihang/projects/tools/Mailbox.py --message uauc reflect;
echo "Forward" >> $2/checkpoint_bestuauc.txt;
CUDA_VISIBLE_DEVICES=$1 WORLD_SIZE=1 torchrun --nproc-per-node 1 --master_port=11189 train_collm_mf_din.py  --cfg-path train_configs/myconfig/reason_mf_book_eval2.yaml >> $2/checkpoint_bestuauc.txt 2>&1;
rm $2/checkpoint_best.pth;
python /home/yuqihang/projects/tools/Mailbox.py --message uauc forward;