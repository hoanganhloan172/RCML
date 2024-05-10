#!/bin/bash

echo "Running CCML"
cd ./../../code/

seed=1
base=0
batch_size=256
epoch=100
loss_fn=BCE
arch=resnet
dataset_path=/media/storagecube/aksoy/ireland12
channel=10
label=BEN-12
prediction_threshold=0.35
sigma=10000
swap=1
swap_start=60
swap_end=60
lambda2=1.
lambda3=.1
miss_alpha=.2
extra_beta=.8
sample_rate=0.4
class_rate=0.4
metric=mmd
test_=0
gpu=1
logname=BENccml40seed1
random_instead_lasso=0

mkdir -p ../output/logs/${logname}

echo "
`date`
seed $seed
batch_size $batch_size
epoch $epoch 
loss_fn $loss_fn
arch $arch 
dataset_path $dataset_path 
channel $channel 
prediction_threshold $prediction_threshold
label $label 
sigma $sigma 
swap $swap 
lambda2 $lambda2 
lambda3 $lambda3 
miss_alpha $miss_alpha 
extra_beta $extra_beta 
sample_rate $sample_rate 
class_rate $class_rate 
metric $metric 
test_ $test_ 
gpu $gpu
logname $logname
swap_start $swap_start
swap_end $swap_end
base $base
random_instead_lasso $random_instead_lasso
" > ../output/logs/${logname}/parameters.txt

python3 main.py --random_instead_lasso $random_instead_lasso --seed $seed --base $base -b $batch_size -e $epoch -l $loss_fn -a $arch -d $dataset_path -ch $channel -prediction_threshold $prediction_threshold -lb $label -si $sigma -sw $swap -lto $lambda2 -ltr $lambda3 -ma $miss_alpha -eb $extra_beta -sar $sample_rate -car $class_rate -dm $metric -test $test_ -gpu $gpu -logname $logname -swap_start $swap_start -swap_end $swap_end > ../output/logs/${logname}/training_log.txt

mkdir -p ../output/logs/${logname}/code

cat evaluate.py > ../output/logs/${logname}/code/evaluate.py
cat loss_fun.py > ../output/logs/${logname}/code/loss_fun.py
cat main.py > ../output/logs/${logname}/code/main.py
cat run.py > ../output/logs/${logname}/code/run.py
cat train.py > ../output/logs/${logname}/code/train.py
cat validate.py > ../output/logs/${logname}/code/validate.py

