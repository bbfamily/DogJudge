# /usr/bin/env sh
DATA=../gen/baidu/image

HASHIQI=${DATA}/哈士奇
LALA=${DATA}/拉布拉多
BM=${DATA}/博美
CQ=${DATA}/柴犬
DM=${DATA}/德国牧羊犬
DB=${DATA}/杜宾

OUTPUT=../gen/dog_judge
mkdir $OUTPUT

echo "Create train.txt..."
rm -rf $OUTPUT/train.txt

find $HASHIQI -name *.jpeg | cut -d '/' -f 5,6 | sed "s/$/ 1/">>$OUTPUT/train.txt
find $LALA -name *.jpeg | cut -d '/' -f 5,6 | sed "s/$/ 2/">>$OUTPUT/lala.txt
find $BM -name *.jpeg | cut -d '/' -f 5,6 | sed "s/$/ 3/">>$OUTPUT/bm.txt
find $CQ -name *.jpeg | cut -d '/' -f 5,6 | sed "s/$/ 4/">>$OUTPUT/cq.txt
find $DM -name *.jpeg | cut -d '/' -f 5,6 | sed "s/$/ 5/">>$OUTPUT/dm.txt
find $DB -name *.jpeg | cut -d '/' -f 5,6 | sed "s/$/ 6/">>$OUTPUT/db.txt

cat $OUTPUT/lala.txt>>$OUTPUT/train.txt
cat $OUTPUT/bm.txt>>$OUTPUT/train.txt
cat $OUTPUT/cq.txt>>$OUTPUT/train.txt
cat $OUTPUT/dm.txt>>$OUTPUT/train.txt
cat $OUTPUT/db.txt>>$OUTPUT/train.txt

rm -rf $OUTPUT/lala.txt
rm -rf $OUTPUT/bm.txt
rm -rf $OUTPUT/cq.txt
rm -rf $OUTPUT/dm.txt
rm -rf $OUTPUT/db.txt

echo "train.txt Done.."