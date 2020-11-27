set -e
cd ../
DATA=cifar100; 
M=kresnet18; 
WDECAY1=.001; 
WDECAY2=.001; 
CUDA=1; 
SD=1; 
AUG=True;
IN_CH=3;
BATCH=128;
ACT=reapen;
HOBJ=srs_upper_tri_alignment;
LOSS=xe;
TR_SIZE=45000
V_SIZE=5000
SDIR=./checkpoints/cifar100_basic/${LOSS}/mdlr_nTrain${TR_SIZE}_nVal${V_SIZE}

CUDA_VISIBLE_DEVICES=${CUDA} python kernet/examples/modular_train.py --dataset ${DATA} --model ${M} --n_parts 2 --loss ${LOSS} --lr1 .1 --lr2 .1 --activation ${ACT} --optimizer sgd --weight_decay1 ${WDECAY1} --weight_decay2 ${WDECAY2} --seed ${SD} --print_freq 1 --n_epochs1 200 --n_epochs2 0 --hidden_objective ${HOBJ} --loglevel info --augment_data ${AUG} --in_channels ${IN_CH} --batch_size ${BATCH}  --save_dir ${SDIR}_1_1 --n_val ${V_SIZE} --max_trainset_size ${TR_SIZE}; 

CUDA_VISIBLE_DEVICES=${CUDA} python kernet/examples/modular_train.py --load_opt --opt_file ${SDIR}_1_1/opt.pkl --save_dir ${SDIR}_1_2 --load_model --checkpoint_dir ${SDIR}_1_1 --lr1 .01 --n_epochs1 100 --dataset_rand_idx ${SDIR}_1_1/dataset_rand_idx.pkl; 

CUDA_VISIBLE_DEVICES=${CUDA} python kernet/examples/modular_train.py --load_opt --opt_file ${SDIR}_1_2/opt.pkl --save_dir ${SDIR}_1_3 --load_model --checkpoint_dir ${SDIR}_1_2 --lr1 .001 --n_epochs1 50; 

CUDA_VISIBLE_DEVICES=${CUDA} python kernet/examples/modular_train.py --load_opt --opt_file ${SDIR}_1_3/opt.pkl --n_epochs1 0 --n_epochs2 20 --save_dir ${SDIR}_2_1 --load_model --checkpoint_dir ${SDIR}_1_3 --lr2 .1;
