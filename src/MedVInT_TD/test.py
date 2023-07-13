import argparse
import os
import json
import math
import tqdm.auto as tqdm
from typing import Optional
import transformers
from Dataset.PMC_QA_Dataset import PMC_QA_Dataset
from models.QA_model import QA_model
from transformers import Trainer
from dataclasses import dataclass, field
import os
from torch.utils.data import DataLoader  
import torch
import numpy as np  
import difflib 
import csv
@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="./LLAMA/llama-7b-hf")
    ckp: Optional[str] = field(default="./Results/QA_PMC_LLaMA_lora_PMC-CLIP_MLP/choice_training/checkpoint-4146")
    checkpointing: Optional[bool] = field(default=False)
    ## Q_former ##
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=8)
    img_token_num: Optional[int] = field(default=32)
    
    ## Basic Setting ##
    voc_size: Optional[int] = field(default=32000)
    hidden_dim: Optional[int] = field(default=4096)
    
    ## Image Encoder ##
    Vision_module: Optional[str] = field(default='PMC-CLIP')
    visual_model_path: Optional[str] = field(default='./img_checkpoint/PMC-CLIP/checkpoint.pt')
    
    ## Peft ##
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)

@dataclass
class DataArguments:
    img_dir: str = field(default='', metadata={"help": "Path to the training data."})
    Test_csv_path: str = field(default='./Data/final_train/final_test.csv', metadata={"help": "Path to the training data."})
    tokenizer_path: str = field(default='./LLAMA/tokenizer', metadata={"help": "Path to the training data."})
    trier: int = field(default=0)
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./Results")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()
 
def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index
  
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Data")
    row_count = 0
    # if os.path.exists('result_final'+str(data_args.trier)+'.csv'): 
        
    #     with open('result_final'+str(data_args.trier)+'.csv', 'r') as file:
    #         reader = csv.reader(file)
    #         row_count = sum(1 for row in reader)-1      
    Test_dataset = PMC_QA_Dataset(data_args.img_dir, data_args.Test_csv_path, data_args.tokenizer_path,mode='Test',start=row_count)
    
    # batch size should be 1
    Test_dataloader = DataLoader(
            Test_dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
    )  

    print("Setup Model")
    ckp = model_args.ckp + '/pytorch_model.bin'
    print(ckp)
    model = QA_model(model_args)
    model.load_state_dict(torch.load(ckp, map_location='cpu'))
    
    ACC = 0
    cc = 0
    model = model.to('cuda')
    model.eval()
    #Test_dataset.tokenizer.padding_side = "left" 
    
    with open('result_final_'+model_args.ckp.split('/')[-3]+'_'+ model_args.ckp.split('/')[-2]+'.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Figure_path','Pred','Label','Correct'])
            for sample in tqdm.tqdm(Test_dataloader):
                input_ids = Test_dataset.tokenizer(sample['input_ids'],return_tensors="pt").to('cuda')
                #input_ids['input_ids'][0][0]=1
                images = sample['images'].to('cuda')
                with torch.no_grad():
                    generation_ids = model.generate(input_ids['input_ids'],images)
                generated_texts = Test_dataset.tokenizer.batch_decode(generation_ids.argmax(-1), skip_special_tokens=True) 
                for i in range(len(generated_texts)):
                    label = sample['labels'][i]
                    img_path = sample['img_path'][i]
                    Choice_A = sample['Choice_A'][i]
                    Choice_B = sample['Choice_B'][i]
                    Choice_C = sample['Choice_C'][i]
                    Choice_D = sample['Choice_D'][i]
                    Choice_list = [Choice_A, Choice_B, Choice_C, Choice_D]
                   
                    pred = generated_texts[i][-1]
                    index_pred = find_most_similar_index(Choice_list,pred)
                    index_label  = find_most_similar_index(Choice_list,label)
                    corret = 0
                    if index_pred == index_label:
                        ACC = ACC +1
                        corret = 1 
                    writer.writerow([img_path,sample['input_ids'][0]+pred,label,corret])
                    cc = cc + 1        
    print(ACC/cc)  
      
if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()
    
#CUDA_VISIBLE_DEVICES=2  python test.py
