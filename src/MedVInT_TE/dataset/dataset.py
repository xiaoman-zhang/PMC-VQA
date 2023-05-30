import csv
import json
import logging
import os
import re
import difflib
import sys
import torch
import random
from abc import abstractmethod
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image
from dataset.randaugment import RandomAugment
from transformers import AutoModel,BertConfig,AutoTokenizer,AutoProcessor,LlamaTokenizer


class Binary_VQA_Dataset(Dataset):
    def __init__(self,csv_path,img_root_dir,image_res,is_blank=True,is_train=True):
        self.is_blank = is_blank
        self.is_train = is_train
        self.root_dir = img_root_dir
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info['Figure_path'])
        self.question_list = np.asarray(data_info['Question'])
        self.choice_list = np.asarray(data_info.iloc[:,-5:-1])
        self.answer_list = np.asarray(data_info['Answer'])
        self.answer_label_list = np.asarray(data_info['Answer_label'])
        self.tokenizer = LlamaTokenizer.from_pretrained('../../LLAMA_Model/tokenizer')
        special_tokens_dict = {'mask_token': "</s>",
                               'eos_token': "</s>",
                               'bos_token': "<s>",
                               'unk_token': "<unk>"}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.pad_token_id=0
        
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop([image_res,image_res],scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','Equalize','Brightness','Sharpness',#AutoContrast',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])   
        
    def encode_mlm(self,question_text,question_text_with_answer,mask_token= '</s>', pad_token='<unk>', eos_token = '</s>'):
        def measure_word_len(word):
            token_ids = self.tokenizer.encode(word)
            # tokens = [tokenizer.decode(x) for x in token_ids]
            return len(token_ids) - 1
        
        question_text_with_answer_tokens = question_text_with_answer.split()
        question_text_tokens = question_text.split()
        bert_input_tokens = []
        output_mask = []
        bert_label_tokens = []  # 被 mask 的保留原词, 否则用 [PAD] 代替
        
        for i, token in enumerate(question_text_with_answer_tokens):
            if i < len(question_text_tokens):
                word_len = measure_word_len(token)
                bert_input_tokens += [token]
                bert_label_tokens += [pad_token]*word_len
                output_mask += [0]*word_len
            else:
                word_len = measure_word_len(token)
                bert_input_tokens += [mask_token]*word_len
                bert_label_tokens += [token]
                output_mask += [1]*word_len
        bert_input_tokens += [eos_token]
        bert_label_tokens += [eos_token]
        bert_input = ' '.join(bert_input_tokens)
        bert_label = ' '.join(bert_label_tokens)
        return bert_input,bert_label
    
    def __len__(self):
        return len(self.img_path_list)
    
    
    def __getitem__(self, index):
        file_name = self.img_path_list[index]
        img_path = os.path.join(self.root_dir,file_name)
        image = Image.open(img_path).convert('RGB')   
        image = self.transform(image)
        
        question = self.question_list[index]
        answer = self.answer_list[index]
        answer_label = self.answer_label_list[index]
        choice = self.choice_list[index]
        Choice_A = str(self.choice_list[index][0])
        Choice_B = str(self.choice_list[index][1])
        Choice_C = str(self.choice_list[index][2])
        Choice_D = str(self.choice_list[index][3])
        if self.is_blank:
            question_text = 'Question: '+ question +'The Answer is:'
            question_text_with_answer = 'Question: '+ question +'The Answer is:' + answer
        else:
            question_text = 'Question: '+ question + 'The choices are: '  + Choice_A + ' ' + Choice_B + ' ' + Choice_C + ' ' + Choice_D + 'The Answer is: '
            question_text_with_answer = 'Question: '+ question + 'The choices are: '  + Choice_A + ' ' + Choice_B + ' ' + Choice_C + ' ' + Choice_D + 'The Answer is: ' + answer_label
        bert_input, bert_label = self.encode_mlm(question_text,question_text_with_answer)

        if self.is_train:
            encoded_input = self.tokenizer(bert_input, add_special_tokens=True, padding='max_length', truncation=True, max_length= 256)
            encoded_label = self.tokenizer(bert_label, add_special_tokens=True, padding='max_length', truncation=True, max_length= 256)
        else:
            encoded_input = self.tokenizer(bert_input, add_special_tokens=True, padding='max_length', truncation=True, max_length= 256,return_tensors="pt")
            encoded_label = self.tokenizer(bert_label, add_special_tokens=True, padding='max_length', truncation=True, max_length= 256,return_tensors="pt")
        

        return {
            "image": image,
            "encoded_input_ids": encoded_input['input_ids'],
            "encoded_attention_mask": encoded_input['attention_mask'],
            "label": encoded_label['input_ids'],
            "image_path": img_path,
            'Choice_A': Choice_A,
            'Choice_B': Choice_B,
            'Choice_C': Choice_C,
            'Choice_D': Choice_D,
            'Answer': answer,
            'Answer_label': answer_label,
            }
        

