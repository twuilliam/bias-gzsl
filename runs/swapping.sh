cd ..

DATA_DIR=data
EXP_DIR=exp/swap

python3 train.py \
    --dataset=cub \
    --temp=0.05 \
    --batch-size=64 \
    --lr=0.01 \
    --exp-dir=$EXP_DIR \
    --data-dir=$DATA_DIR \
    --epochs=15 \
    --seed=456 \
    --mlp \
    --normalize \
    --m=attributes

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
    --sentences \
    --m=sentences

python3 extract.py --model-path=$EXP_DIR/cub_attributes/checkpoint.pth.tar
python3 extract.py --model-path=$EXP_DIR/cub_sentences/checkpoint.pth.tar

python3 swapping.py \
    --att=$EXP_DIR/cub_attributes/feat.npz \
    --sen=$EXP_DIR/cub_sentences/feat.npz
