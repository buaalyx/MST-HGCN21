date=$(date "+%Y-%m-%d")
dataset="pubmed"
mkdir -p att_result_tmp/$dataset/$date
mkdir -p att_result_tmp/$dataset/$date/tune

for num in $(seq 0 9)
do
	python train_raw.py --dataset pubmed --dropout 0.6048173167494098 --weight-decay 0.003 --lr 0.0021213442368275557 --lr-reduce-freq 250 --patience 105 --spt-alg kruskal --spt-attr similarity --model HGCN --manifold Hyperboloid --num-layers 3 --dim 160 --num-adj 4 --add-rate 0.45 --tem 1.1452812127698557 --consis-rate 1 --c None --seed $num > att_result_tmp/$dataset/"$date"/"$num".txt
done

python 100.py $dataset 10 att_result_tmp/$dataset/$date/
