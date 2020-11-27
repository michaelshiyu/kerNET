set -e
cd ../
DATA=mnist; 
M=k1lenet5; 
WDECAY=.0002; 
CUDA=0; 
SD=3 
AUG=False;
IN_CH=1;
BATCH=128;
TR_SIZE=55000
V_SIZE=5000
ACT=relu;
LOSS=xe;
ASAVE=False
SDIR=./checkpoints/mnist_basic/${LOSS}/seed${SD}/e2e_nTrain${TR_SIZE}_nVal${V_SIZE}

CUDA_VISIBLE_DEVICES=${CUDA} python kernet/examples/train.py --dataset ${DATA} --model ${M} --loss ${LOSS} --lr .1 --activation ${ACT} --optimizer sgd --weight_decay ${WDECAY} --seed ${SD} --print_freq 1 --n_epochs 200 --loglevel info --n_classes 10 --augment_data ${AUG} --in_channels ${IN_CH} --batch_size ${BATCH}  --save_dir ${SDIR}_1 --max_trainset_size ${TR_SIZE} --n_val ${V_SIZE} --always_save ${ASAVE}; 

CUDA_VISIBLE_DEVICES=${CUDA} python kernet/examples/train.py --load_opt --opt_file ${SDIR}_1/opt.pkl --save_dir ${SDIR}_2 --load_model --checkpoint_dir ${SDIR}_1 --lr .01 --n_epochs 100 --dataset_rand_idx ${SDIR}_1/dataset_rand_idx.pkl 

CUDA_VISIBLE_DEVICES=${CUDA} python kernet/examples/train.py --load_opt --opt_file ${SDIR}_2/opt.pkl --save_dir ${SDIR}_3 --load_model --checkpoint_dir ${SDIR}_2 --lr .001 --n_epochs 50 
