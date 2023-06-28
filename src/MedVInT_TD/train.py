import argparse
import os
import json
import math
import tqdm.auto as tqdm
from typing import Optional
import transformers
from Dataset.PMC_QA_Dataset import PMC_QA_Dataset
from model.QA_model import QA_model
from transformers import Trainer
from dataclasses import dataclass, field
import os
from torch.utils.data import DataLoader  
import torch
import wandb      
@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="./LLAMA/checkpoint-189000/")
    ## Q_former ##
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=12)
    img_token_num: Optional[int] = field(default=32)
    
    ## Basic Setting ##
    voc_size: Optional[int] = field(default=32000)
    hidden_dim: Optional[int] = field(default=4096)
    checkpointing: Optional[bool] = field(default=True)
    ## Image Encoder ##
    Vision_module: Optional[str] = field(default='CLIP')
    visual_model_path: Optional[str] = field(default='./img_checkpoint/CLIP/clip-vit-base-patch32')
    #visual_model_config: Optional[str] = field(default='./img_checkpoint/RN50_fusion4.json')
    ## Peft ##
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)

@dataclass
class DataArguments:
    img_dir: str = field(default='./PMC_OA_papers/figures/', metadata={"help": "Path to the training data."})
    pred_type: str = field(default='choice')
    Train_csv_path: str = field(default='./Data/final_train/final_train.csv', metadata={"help": "Path to the training data."})
    Eval_csv_path: str = field(default='./Data/final_train/final_test.csv', metadata={"help": "Path to the training data."})
    tokenizer_path: str = field(default='./LLAMA/tokenizer', metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    output_dir: Optional[str] = field(default="./Results")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    
    
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Data")
    Train_dataset = PMC_QA_Dataset(data_args.img_dir, data_args.Train_csv_path, data_args.tokenizer_path, text_type = 'caption')
    Eval_dataset = PMC_QA_Dataset(data_args.img_dir, data_args.Eval_csv_path, data_args.tokenizer_path, text_type = 'caption')

    print("Setup Model")
    model = QA_model(model_args)
    
    run_name_root = training_args.run_name
    output_dir_root = training_args.output_dir
    
    training_args.run_name = run_name_root+'_caption_pretrain'
    training_args.output_dir = output_dir_root + '/caption_pretrain/'
    
    print('Start Pretraining')
    trainer = Trainer(model=model, 
                      train_dataset = Train_dataset, 
                      eval_dataset = Eval_dataset,
                      args=training_args,
                      )
    try:
        trainer.train(resume_from_checkpoint=True)
    except:
        trainer.train()
    trainer.save_state()
    
    print('Start training')  
    
    training_args.run_name = run_name_root+'_' + data_args.pred_type + '_training'
    training_args.output_dir = output_dir_root + '/'+data_args.pred_type +'_training/'
    
    Train_dataset.text_type = data_args.pred_type
    Eval_dataset.text_type = data_args.pred_type
    
    trainer = Trainer(model=model, 
                      train_dataset = Train_dataset, 
                      eval_dataset = Eval_dataset,
                      args=training_args,
                      )
    # try:
    #     trainer.train(resume_from_checkpoint=True)
    # except: 
    trainer.train(resume_from_checkpoint=True)
    
    trainer = Trainer(model=model, 
                      train_dataset = Train_dataset, 
                      eval_dataset = Eval_dataset,
                      args=training_args,
                      )
    
    trainer.train()
    trainer.save_state()
    
if __name__ == "__main__":
    main()