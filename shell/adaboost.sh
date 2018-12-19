source activate fml
cd /home/chenlei/facial_expression_recognition/src
#lr, dp, num, algo

for DP in 15 25
do
for NUM in 1 5 10 50 100 300 600
do
for ALGO in SAMME
do
for LR in 0.1 0.5
do
current_time="`date +%d%H%M%S`"

nohup python3 adaboost.py \
--n_estimators ${NUM} \
--depth ${DP} \
--algo ${ALGO} \
--lr ${LR} \
2>&1 | tee ../output/adaboost_${NUM}num_${DP}dpth_${ALGO}algo_${LR}lr_${current_time}.output \
&
done
done
done
done

wait

for DP in 15 25
do
for NUM in 200 400
do
for ALGO in SAMME
do
for LR in 0.0001 0.0001 0.001 0.01 0.1 0.5 1
do
current_time="`date +%d%H%M%S`"

nohup python3 adaboost.py \
--n_estimators ${NUM} \
--depth ${DP} \
--algo ${ALGO} \
--lr ${LR} \
2>&1 | tee ../output/adaboost_${NUM}num_${DP}dpth_${ALGO}algo_${LR}lr_${current_time}.output \
&
done
done
done
done

wait

for DP in  15 25
do
for NUM in 200 400
do
for ALGO in SAMME SAMME.R
do
for LR in 0.1 0.5
do
current_time="`date +%d%H%M%S`"

nohup python3 adaboost.py \
--n_estimators ${NUM} \
--depth ${DP} \
--algo ${ALGO} \
--lr ${LR} \
2>&1 | tee ../output/adaboost_${NUM}num_${DP}dpth_${ALGO}algo_${LR}lr_${current_time}.output \
&
done
done
done
done

wait

for DP in 1 2 5 10 15 20 25
do
for NUM in 200 400
do
for ALGO in SAMME
do
for LR in 0.1 0.5
do
current_time="`date +%d%H%M%S`"

nohup python3 adaboost.py \
--n_estimators ${NUM} \
--depth ${DP} \
--algo ${ALGO} \
--lr ${LR} \
2>&1 | tee ../output/adaboost_${NUM}num_${DP}dpth_${ALGO}algo_${LR}lr_${current_time}.output \
&
done
done
done
done

