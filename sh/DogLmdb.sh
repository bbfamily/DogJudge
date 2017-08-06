#!/usr/bin/env bash
# /usr/bin/env sh
echo "Begin lmdb..."
ROOTFOLDER=../gen/baidu/image/
OUTPUT=../gen/dog_judge
CONVERT_BIN=/root/caffe/build/tools/convert_imageset

rm -rf $OUTPUT/img_train_lmdb
$CONVERT_BIN --shuffle --resize_height=256 --resize_width=256 $ROOTFOLDER $OUTPUT/train_split.txt  $OUTPUT/img_train_lmdb

rm -rf $OUTPUT/img_val_lmdb
$CONVERT_BIN --shuffle --resize_height=256 --resize_width=256 $ROOTFOLDER $OUTPUT/val_split.txt  $OUTPUT/img_val_lmdb
echo "Done lmdb..."


