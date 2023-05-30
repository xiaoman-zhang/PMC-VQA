import argparse
import os
import json
import math
import numpy as np
import tqdm.auto as tqdm
from typing import Optional
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
import os
from torch import nn
from torch.utils.data import DataLoader  
from torch.nn import functional as F
import torch
from tensorboardX import SummaryWriter

from dataset.dataset import Binary_VQA_Dataset 
from models.llama.vqa_model import Binary_VQA_Model

class VQATrainer(Trainer):
    # 这里没问题
    def compute_loss(self, model, inputs, return_outputs=False): ## compute loss这个步骤实际上定义了 forward和loss的计算过程。。。
        image = inputs['image']  
        label = inputs['labels'].to(dtype=torch.long) 
        question_inputids = inputs['encoded_input_ids'] 
        question_attenmask = inputs['encoded_attention_mask'] 
        outputs = model(image,question_inputids,question_attenmask)
        loss = F.nll_loss(outputs.transpose(1, 2), label, ignore_index=0)
        return (loss, {'outputs':outputs}) if return_outputs else loss #outputs要用字典返回 


@dataclass
class ModelArguments:
    embed_dim: Optional[int] = field(default=768)
    pretrained_tokenizer:  Optional[str] = field(default="../../LLAMA_Model/tokenizer")
    pretrained_model: Optional[str] = field(default="../../LLAMA_Model/llama-7b-hf")
    image_encoder: Optional[str] = field(default="CLIP")
    pmcclip_pretrained: Optional[str] = field(default="./models/pmc_clip/checkpoint.pt")
    clip_pretrained: Optional[str] = field(default="openai/clip-vit-base-patch32")
    ckp: Optional[str] = field(default="./Results/VQA_lora_noclip/vqa/checkpoint-6500")


@dataclass
class DataArguments:
    is_blank: Optional[bool] = field(default=False)
    image_res: Optional[int] = field(default=512)
    img_root_dir: str = field(default='../../PMC-VQA/images/', metadata={"help": "Path to the training data."})
    Train_csv_path: str = field(default= '../../PMC-VQA/train.csv', metadata={"help": "Path to the training data."})
    Test_csv_path: str = field(default= '../../PMC-VQA/test.csv', metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./Results")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    logging_dir: Optional[str] = field(default="./logs")
    logging_steps: Optional[int] = field(default=50)
    

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Data")
    Train_dataset = Binary_VQA_Dataset(data_args.Train_csv_path, data_args.img_root_dir, data_args.image_res,model_args.pretrained_tokenizer,is_blank=data_args.is_blank,is_train=True)
    Eval_dataset = Binary_VQA_Dataset(data_args.Test_csv_path, data_args.img_root_dir, data_args.image_res,model_args.pretrained_tokenizer,is_blank=data_args.is_blank,is_train=True)

    print("Setup Model")
    model = Binary_VQA_Model(model_args)
    
    run_name_root = training_args.run_name
    output_dir_root = training_args.output_dir
    

    print('Start training')  
    
    training_args.run_name = run_name_root+'_vqa'
    training_args.output_dir = output_dir_root + '/vqa/'
    
    
    trainer = VQATrainer(model=model, 
                      train_dataset = Train_dataset, 
                      eval_dataset = Eval_dataset,
                      args=training_args
                      )
    trainer.train()
    trainer.save_state()

    
if __name__ == "__main__":
    main()