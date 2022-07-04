date=$(date "+%Y-%m-%d")
dataset="airport"
mkdir -p att_result_tmp/$dataset/$date
mkdir -p att_result_tmp/$dataset/$date/tune

for num in $(seq 0 9)
do
	python train_raw.py --dataset airport --dropout 0 --weight-decay 0.0004 --lr 0.01 --lr-reduce-freq 180 --patience 150 --spt-alg prim --spt-attr None --model HGCN --manifold PoincareBall --num-layers 3 --dim 256 --num-adj 8 --add-rate 0.8 --tem 0.5 --consis-rate 1 --c None --seed $num > att_result_tmp/$dataset/"$date"/"$num".txt
done

python 100.py $dataset 10 att_result_tmp/$dataset/$date/
