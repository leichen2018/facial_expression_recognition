source activate fml
cd /home/chenlei/facial_expression_recognition/src

for DP in 15 25
do
for NUM in 1 50 100 200 300 400 500 600 700 800 900 1000 1100
do
current_time="`date +%d%H%M%S`"

nohup python3 randforest.py \
--n_estimators ${NUM} \
--depth ${DP} \
2>&1 | tee ../output/randforest_${NUM}num_${DP}dpth_${current_time}.output \
&
done
done

wait

for DP in 1 2 5 10 15 20 25 50
do
for NUM in 500 1100
do
current_time="`date +%d%H%M%S`"

nohup python3 randforest.py \
--n_estimators ${NUM} \
--depth ${DP} \
2>&1 | tee ../output/randforest_${NUM}num_${DP}dpth_${current_time}.output \
&
done
done
