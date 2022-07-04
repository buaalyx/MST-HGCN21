date=$(date "+%Y-%m-%d")
dataset="citeseer"
mkdir -p att_result_tmp/$dataset/$date
mkdir -p att_result_tmp/$dataset/$date/tune

for num in $(seq 0 9)
do
	python train_raw.py --dataset citeseer --dropout 0.42215067338671464 --weight-decay 0.0012074788677796655 --lr 0.002269488941889155 --lr-reduce-freq 180 --patience 130 --spt-alg kruskal --spt-attr None --model HGCN --manifold PoincareBall --num-layers 3 --dim 332 --num-adj 6 --add-rate 0.03 --tem 1.0 --consis-rate 1.3 --c None --use-att 0 --seed $num > att_result_tmp/$dataset/"$date"/"$num".txt
done

python 100.py $dataset 10 att_result_tmp/$dataset/$date/
