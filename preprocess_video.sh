src='ja'
tgt='en'

TEXT=data/opus-$src-$tgt

# In text, each line is a sentence
rm -rf /home/yihang/code/mmt_model_rebuild/fairseq_mmt/data-bin/opus.ja-en

fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test \
  --destdir data-bin/opus.$src-$tgt \
  --thresholdtgt 3 \
  --thresholdsrc 3 \
  --workers 8 

