cd ..

DATA_DIR=data
FEAT_DIR=feat/cyclegan
EXP_DIR=exp/cyclegan

python3 train.py \
    --dataset=cub \
    --temp=0.05 \
    --batch-size=64 \
    --lr=0.01 \
    --exp-dir=$EXP_DIR \
    --gan-path=$FEAT_DIR/cub/feat.npz \
    --data-dir=$DATA_DIR \
    --epochs=25 \
    --seed=456 \
    --mlp \
    --nhidden=2048 \
    --sentences \
    --ent=0.1 \
    --margin=0.2

python3 train.py \
    --dataset=sun \
    --temp=0.05 \
    --batch-size=64 \
    --lr=0.01 \
    --exp-dir=$EXP_DIR \
    --gan-path=$FEAT_DIR/sun/feat.npz \
    --data-dir=$DATA_DIR \
    --epochs=25 \
    --seed=456 \
    --mlp \
    --nhidden=1024 \
    --ent=0.1 \
    --margin=0.2

python3 train.py \
    --dataset=awa1 \
    --temp=0.05 \
    --batch-size=64 \
    --lr=0.0001 \
    --exp-dir=$EXP_DIR \
    --gan-path=$FEAT_DIR/awa1/feat.npz \
    --data-dir=$DATA_DIR \
    --epochs=5 \
    --seed=456 \
    --mlp \
    --nhidden=1024 \
    --ent=0.5 \
    --margin=0.2 \
    --no-decay

python3 train.py \
    --dataset=flo \
    --temp=0.05 \
    --batch-size=64 \
    --lr=0.01 \
    --exp-dir=$EXP_DIR \
    --gan-path=$FEAT_DIR/flo/feat.npz \
    --data-dir=$DATA_DIR \
    --epochs=50 \
    --seed=456 \
    --mlp \
    --nhidden=2048 \
    --ent=0.1 \
    --margin=0.2
