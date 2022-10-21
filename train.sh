CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
--config-file configs/train_alphanet_models_add.yml > adder_log/adder_cifar100_pro_pro_noconstraint 2>&1 &