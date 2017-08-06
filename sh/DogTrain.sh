# /usr/bin/env sh
echo "Begin train...."

CAFEBIN=/root/caffe/build/tools/caffe
SOLVER=../pb/solver.prototxt
$CAFEBIN train -solver $SOLVER


echo "Done train..."
