set -e
set -x
export PYTHONPATH='src'


nepoch=1
batchSize=32
num_workers=12
lr=1e-3
cuda=0

CUDA_VISIBLE_DEVICES=$cuda python -u test.py \
    --nepoch=$nepoch \
    --batchSize=$batchSize \
    --num_workers=$num_workers \
    --lr=$lr \
    --cuda=$cuda \
