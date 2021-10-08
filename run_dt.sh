question=$1
train_data=$2
val_data=$3
test_data=$4
output_file=$5

if [ ${question} == "1" ] ; then
	python3 Q1/dtree.py $train_data $val_data $test_data $output_file
elif [ ${question} == "2" ] ; then
	python3 Q1/pruneddtree.py $train_data $val_data $test_data $output_file
fi