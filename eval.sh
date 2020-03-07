#!/bin/bash
 
DATA_TRAIN=$ARNOLD_TRAIN
DATA_VAL=/mnt/cephfs_new_wj/$ARNOLD_CEPH_TEST
OUTPUT=$ARNOLD_OUTPUT
MODEL_DIR=/mnt/cephfs_new_wj/$ARNOLD_CEPH_PREMODEL
cd $(dirname $0)

for name in `ls $MODEL_DIR`
do
  if [ "${name##*.}"x = "pt"x ]; then
    echo $name
    if test -f $MODEL_DIR/predict_$name
    then
        echo '文件已存在!'
    else
        python3 translate.py --src_dico_file $DATA_VAL/src_dico_file --tgt_dico_file $DATA_VAL/tgt_dico_file --translate_file $DATA_VAL/translate_file --reference_file $DATA_VAL/reference_file --checkpoint_dir $MODEL_DIR --model_name $name $@
        echo $name >> $MODEL_DIR/bleu.log
        perl scripts/multi-bleu.perl $DATA_VAL/reference_file < $MODEL_DIR/predict_${name:0:-3} | grep BLEU >> $MODEL_DIR/bleu.log 
    fi
  fi
done
