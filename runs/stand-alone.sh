cd ..

DATA_DIR=data
EXP_DIR=exp/raw

python3 train.py \
    --dataset=cub \
    --temp=0.05 \
    --batch-size=64 \
    --lr=0.01 \
    --exp-dir=$EXP_DIR \
    --data-dir=$DATA_DIR \
    --epochs=20 \
    --seed=456 \
    --mlp \
    --normalize \
    --nhidden=2048 \
    --sentences

python3 train.py \
    --dataset=awa1 \
    --temp=0.05 \
    --batch-size=64 \
    --lr=0.001 \
    --exp-dir=$EXP_DIR \
    --data-dir=$DATA_DIR \
    --epochs=5 \
    --seed=456 \
    --mlp \
    --normalize

python3 train.py \
    --dataset=sun \
    --temp=0.05 \
    --batch-size=64 \
    --lr=0.01 \
    --exp-dir=$EXP_DIR \
    --data-dir=$DATA_DIR \
    --epochs=15 \
    --seed=456 \
    --mlp \
    --normalize

python3 train.py \
    --dataset=flo \
    --temp=0.05 \
    --batch-size=64 \
    --lr=0.01 \
    --exp-dir=$EXP_DIR \
    --data-dir=$DATA_DIR \
    --epochs=15 \
    --seed=456 \
    --nhidden=2048 \
    --mlp \
    --normalize
