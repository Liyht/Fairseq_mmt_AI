#!/usr/bin/bash
set -e

_mask=mask0
_image_feat=$1

# set device
gpu=2

model_root_dir=checkpoints

# set task
task=opvi-ja2en
mask_data=$_mask
image_feat=$_image_feat

who=test
random_image_translation=0 #1
length_penalty=0.8

# set tag
model_dir_tag=$image_feat/$image_feat-$mask_data

if [ $task == 'opvi-ja2en' ]; then
	src_lang=ja
	tgt_lang=en
	data_dir=opvi.ja-en
fi


if [[ $image_feat == *"i3d"* ]] ; then
# 	image_feat_path=/mnt/zamia/li/dataset/OpusEJ/OpusEJ_i3d_feature
	image_feat_path=/data/yihang/AS_pa_i3d_feature
	image_feat_dim=2048 # (32, 2048)
fi

# data set
ensemble=10
batch_size=400
beam=10
src_lang=ja

model_dir=$model_root_dir/$task/$model_dir_tag

checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi

output=$model_dir/translation_$who.log

export CUDA_VISIBLE_DEVICES=$gpu

cmd="fairseq-generate data-bin/$data_dir 
  -s $src_lang -t $tgt_lang 
  --path $model_dir/$checkpoint 
  --gen-subset $who 
  --batch-size $batch_size --beam $beam --lenpen $length_penalty 
  --quiet --remove-bpe
  --task image_mmt
  --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim
  --output $model_dir/hypo.txt" 

echo ${cmd}
 
if [ $random_image_translation -eq 1 ]; then
cmd=${cmd}" --random-image-translation "
fi

cmd=${cmd}" | tee "${output}
eval $cmd

python3 rerank.py $model_dir/hypo.txt $model_dir/hypo.sorted

if [ $task == "multi30k-en2de" ] && [ $who == "test" ]; then
	ref=data/multi30k/test.2016.de
elif [ $task == "multi30k-en2de" ] && [ $who == "test1" ]; then
	ref=data/multi30k/test.2017.de
elif [ $task == "multi30k-en2de" ] && [ $who == "test2" ]; then
	ref=data/multi30k/test.coco.de

elif [ $task == "multi30k-en2fr" ] && [ $who == 'test' ]; then
	ref=data/multi30k/test.2016.fr
elif [ $task == "multi30k-en2fr" ] && [ $who == 'test1' ]; then
	ref=data/multi30k/test.2017.fr
elif [ $task == "multi30k-en2fr" ] && [ $who == 'test2' ]; then
	ref=data/multi30k/test.coco.fr
    
elif [ $task == "opus-ja2en" ] && [ $who == 'test' ]; then
	ref=data/opus-ja-en/test.en
elif [ $task == "opvi-ja2en" ] && [ $who == 'test' ]; then
	ref=data/opvi-ja-en/test.en
fi	

hypo=$model_dir/hypo.sorted
python3 meteor.py $hypo $ref > $model_dir/meteor_$who.log
# echo  "python3 meteor.py $hypo $ref > $model_dir/meteor_$who.log"
cat $model_dir/meteor_$who.log
