# /usr/bin/env sh
echo "Begin..."

LMDB=../gen/dog_judge/img_train_lmdb
MEANBIN=/Users/Bailey/caffe/build/tools/compute_image_mean
OUTPUT=../gen/dog_judge/mean.binaryproto

echo $OUTPUT

$MEANBIN $LMDB $OUTPUT

LMDB=../gen/dog_judge/img_val_lmdb
OUTPUT=../gen/dog_judge/mean_val.binaryproto
echo $OUTPUT
$MEANBIN $LMDB $OUTPUT

echo "Done.."