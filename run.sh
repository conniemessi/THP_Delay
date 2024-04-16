device=0
data=data/toy_one/5dim_5000seq_160ev/
batch=128
n_head=1
n_layers=1
d_model=16
d_rnn=8
d_inner=32
d_k=16
d_v=16
dropout=0.1
lr=0.0001
smooth=0.1
epoch=100
log=data/toy_one/5dim_5000seq_160ev/100_20_input3_hidden16_softplus_batch128_grad.txt
num_events=64
lr_delay=0.0001
n_hidden=64
n_input=5

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log -num_events $num_events -lr_delay $lr_delay -n_hidden $n_hidden -n_input $n_input