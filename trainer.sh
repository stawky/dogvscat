set -e
set -x
export PYTHONPATH='src'


nepoch=40 
batchSize=33
num_workers=24
lr=0.001
cuda=0
optimizer=SGD

CUDA_VISIBLE_DEVICES=$cuda python -u train.py \
    --nepoch=$nepoch \
    --batchSize=$batchSize \
    --num_workers=$num_workers \
    --lr=$lr \
    --cuda=$cuda \
    --optimizer=$optimizer