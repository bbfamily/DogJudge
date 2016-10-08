# /usr/bin/env sh
echo "Begin..."
ROOTFOLDER=../gen/baidu/image/
OUTPUT=../gen/dog_judge

rm -rf $OUTPUT/img_train_lmdb
/Users/Bailey/caffe/build/tools/convert_imageset --shuffle \
--resize_height=256 --resize_width=256 \
$ROOTFOLDER $OUTPUT/train_split.txt  $OUTPUT/img_train_lmdb

rm -rf $OUTPUT/img_val_lmdb
/Users/Bailey/caffe/build/tools/convert_imageset --shuffle \
--resize_height=256 --resize_width=256 \
$ROOTFOLDER $OUTPUT/val_split.txt  $OUTPUT/img_val_lmdb d
echo "Done.."


