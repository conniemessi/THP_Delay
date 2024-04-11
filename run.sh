device=0
data=data/toy/3dim_20000seq_32ev/
batch=32
n_head=1
n_layers=1
d_model=16
d_rnn=8
d_inner=64
d_k=16
d_v=16
dropout=0.1
lr=0.001
smooth=0.1
epoch=100
log=data/toy/3dim_20000seq_32ev/100_25.txt
num_events=32
lr_delay=0.01

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log -num_events $num_events -lr_delay $lr_delay