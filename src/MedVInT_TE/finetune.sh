export PATH=/usr/local/cuda/bin:$PATH
CUDA_VISIBLE_DEVICES=4,5 \
torchrun --nproc_per_node=2 --master_port 19934 finetune.py \
    --bf16 True \
    --output_dir ./Results_finetune/SLAKE \
    --pretrained_model ./PMC_LLAMA_Model  \
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --run_name SLAKE\
    --tf32 True \