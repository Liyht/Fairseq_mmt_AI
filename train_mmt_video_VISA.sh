# mmt_model_rebuild/fairseq_mmt/fairseq/data/image_dataset.py to set feat dataloader

# codes have been edited:
# mmt_model_rebuild/fairseq_mmt/fairseq/data/image_dataset.py
# mmt_model_rebuild/fairseq_mmt/fairseq/tasks/image_multimodal_translation.py
# mmt_model_rebuild/fairseq_mmt/fairseq/data/image_language_pair_dataset.py


# set before using:
# device, image_feat, max_tokens, image_dataset.py->random.shuffle

#! /usr/bin/bash
set -e

# device=0,1,2,3,4,5,6,7
device=0,1,2,3,4,5,6,7
# device=2,3
image_feat=i3d
# image_feat=i3d1
patience=10
max_tokens=8000
# max_tokens=2000 # for videoMAE and i3d
# max_tokens=10000 # for c4c
fp16=0 #0

task=opvi-ja2en
mask_data=mask0
tag=$image_feat/$image_feat-$mask_data
save_dir=checkpoints/$task/$tag

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi

if [ $task == 'opvi-ja2en' ]; then
	src_lang=ja
	tgt_lang=en
	data_dir=opvi.ja-en
fi

criterion=label_smoothed_cross_entropy
amp=0
lr=0.005
warmup=2000
update_freq=1
keep_last_epochs=10
max_update=800000
dropout=0.3

arch=image_multimodal_transformer_SA_top
SA_attention_dropout=0.1
SA_image_dropout=0.1

if [[ $image_feat == *"i3d"* ]] ; then
# 	image_feat_path=/mnt/zamia/li/dataset/OpusEJ/OpusEJ_i3d_feature
	image_feat_path=/data/yihang/AS_pa_i3d_feature
	image_feat_dim=2048 # (32, 2048)
	image_feat_whole_dim="32 2048" # (32, 2048)
fi

# multi-feature
#image_feat_path=data/vit_large_patch16_384 data/vit_tiny_patch16_384
#image_feat_dim=1024 192

cp ${BASH_SOURCE[0]} $save_dir/train.sh

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

# export PYTHONPATH=$PYTHONPATH:/home/yihang/code/mmt_model_rebuild/fairseq_mmt/fairseq
# echo $PYTHONPATH
#   --user-dir fairseq/tasks
#   --share-all-embeddings
#   --num-workers 2
#   --find-unused-parameters
#   --model-parallel-size
#   --load-checkpoint-on-all-dp-ranks
#   --data-buffer-size 1
#   --restore-file checkpoints/opus-ja2en/i3d/i3d-mask0/checkpoint_best.pt
cmd="fairseq-train data-bin/$data_dir
  --save-dir $save_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --dropout $dropout
  --criterion $criterion --label-smoothing 0.1
  --task image_mmt --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim
  --image-feat-whole-dim $image_feat_whole_dim
  --optimizer adam --adam-betas '(0.9, 0.98)'
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --max-tokens $max_tokens --update-freq $update_freq --max-update $max_update
  --patience $patience
  --keep-last-epochs $keep_last_epochs
  --num-workers 2
  --log-interval 20
  --reset-optimizer"

if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi

if [ $amp -eq 1 ]; then
cmd=${cmd}" --amp "
fi

if [ -n "$SA_image_dropout" ]; then
cmd=${cmd}" --SA-image-dropout "${SA_image_dropout}
fi
if [ -n "$SA_attention_dropout" ]; then
cmd=${cmd}" --SA-attention-dropout "${SA_attention_dropout}
fi

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log

# videoMAE_compress 8gpu kubera 2000token --fp16-init-scale 64
