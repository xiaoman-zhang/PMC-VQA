# MedVInT_TE - Official PyTorch Implementation

We provide the official **Pytorch** implementation of training and testing MedVInT_TE with different pre-trained models for both choice and blank tasks

## Usage

### 1. Create Environment 

Please refer to https://github.com/chaoyi-wu/PMC-LLaMA

### 2. Prepare Dataset 

Download from [Huggingface](https://huggingface.co/datasets/xmcmic/PMC-VQA/) and save into ../../PMC-VQA

### 3. Training

sh train.sh

### 4. Finetuning

sh finetune.sh

**Default**: MedVInT-TE-Transformer, LLaMA-ENC, CLIP for Multiple Choice task

Note that to run MedVInT-TE with PMCCLIP, you should first download pmcclip pretrained model from [PMC-CLIP](https://github.com/WeixiongLin/PMC-CLIP), and save to `./models/pmc_clip`

```bash
export PATH=/usr/local/cuda/bin:$PATH

CUDA_LAUNCH_BLOCKING=1 \
srun --partition=your_partition --mpi=pmi2 --gres=gpu:2 -n1 --ntasks-per-node=1  --job-name=VQA_LoRA_training --kill-on-bad-exit=1 \
    torchrun --nproc_per_node=2 --master_port 18832 train.py \
    --bf16 True \
    --output_dir ./Results/VQA_lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --eval_steps 5 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --run_name VQA_LoRA_training \
    --tf32 True \
    # --is_blank True \ if is_blank
    # --deepspeed ./ds_config/ds_config_zero2.json \ if deep_speed
    # --pretrained_model ./PMC_LLAMA_Model  \ if PMC-LLaMA, change this to your PMC-LLaMA model path
    # --image_encoder "PMC_CLIP" \ if PMC-CLIP 
    # --pmcclip_pretrained "./models/pmc_clip/checkpoint.pt" \ if PMC-CLIP, change this to your PMC-CLIP model path

```

### 4. Evaluation

We provide the pre-trained checkpoint of the multiple-choice task of LLaMA_CLIP and LLaMA_PMCCLIP. Download the pre-trained [MedVInT-TE](), and save into `./Results `directly.

Load checkpoint and eval on 2k samples from test_clean.csv.

***\*LLaMA_CLIP\****

srun --partition=your_partition --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1  --job-name=VQA_LoRA_test --kill-on-bad-exit=1 torchrun --nproc_per_node=1 --master_port 12345 test.py --output_dir ./Results/VQA_lora --ckp ./Results/VQA_lora/vqa/checkpoint-6500  

***\*LLaMA_PMCCLIP\****

srun --partition=your_partition --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1  --job-name=VQA_LoRA_test --kill-on-bad-exit=1 torchrun --nproc_per_node=1 --master_port 12345 test.py --output_dir ./Results/VQA_lora_pmcclip --ckp ./Results/VQA_lora_pmcclip/vqa/checkpoint-13500  --image_encoder PMC_CLIP 

## Citation

If you use this code or use our pre-trained weights for your research, please cite our [paper](https://arxiv.org/abs/2305.10415)

```
@article{zhang2023pmcvqa,
      title={PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering}, 
      author={Xiaoman Zhang and Chaoyi Wu and Ziheng Zhao and Weixiong Lin and Ya Zhang and Yanfeng Wang and Weidi Xie},
      year={2023},
      journal={arXiv preprint arXiv:2305.10415},
}
```

