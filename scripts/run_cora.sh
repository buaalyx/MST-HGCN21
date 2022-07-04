date=$(date "+%Y-%m-%d")
dataset="cora"
mkdir -p att_result_tmp/$dataset/$date
mkdir -p att_result_tmp/$dataset/$date/tune

for num in $(seq 0 9)
do
	python train_raw.py --dataset cora --dropout 0.7 --weight-decay 0.0005 --lr 0.005 --lr-reduce-freq 100 --patience 100 --spt-alg kruskal --spt-attr edge_degree --model HGCN --manifold PoincareBall --num-layers 3 --dim 256 --num-adj 4 --add-rate 0.5 --tem 0.89 --consis-rate 1.63 --c None --use-att 1 --seed $num > att_result_tmp/$dataset/"$date"/"$num".txt
done

python 100.py $dataset 10 att_result_tmp/$dataset/$date/