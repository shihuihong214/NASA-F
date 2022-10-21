CUDA_VISIBLE_DEVICES=0 nohup python -u fine_tune.py \
--config-file configs/fine_tune_adder.yml > finetune/propro_finetune 2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -u test.py \
# --config-file configs/fine_tune_adder.yml > search_log/ratio_1stage 2>&1 &